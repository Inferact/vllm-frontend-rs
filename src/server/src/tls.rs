//! TLS support for the HTTP and gRPC servers.
//!
//! Provides a [`TlsListener`] that wraps a `TcpListener` with a `rustls` TLS acceptor,
//! compatible with [`axum::serve`]. Supports automatic certificate reloading via
//! [`spawn_cert_refresh`] when `--enable-ssl-refresh` is set. The same hot-swappable
//! [`ServerConfig`] holder is shared with the gRPC listener via [`tls_incoming`].

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime};

use anyhow::{Context as _, Result, bail};
use arc_swap::ArcSwap;
use futures::Stream;
use rustls::ServerConfig;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio::net::{TcpListener, TcpStream};
use tokio::time;
use tokio_rustls::server::TlsStream;
use tracing::{debug, info, warn};

use crate::config::TlsConfig;

/// A TLS-enabled TCP listener compatible with [`axum::serve`].
///
/// Wraps a `TcpListener` and a hot-swappable `rustls::ServerConfig` so that each accepted
/// connection goes through a TLS handshake. Failed handshakes are logged and retried
/// transparently.
pub struct TlsListener<L> {
    inner: L,
    config: Arc<ArcSwap<ServerConfig>>,
}

impl<L> TlsListener<L> {
    /// Build a listener using a pre-existing shared config holder, so that HTTP and gRPC
    /// paths can share the same hot-swappable TLS state.
    pub fn from_holder(inner: L, config: Arc<ArcSwap<ServerConfig>>) -> Self {
        Self { inner, config }
    }
}

/// TLS handshake timeout. Prevents slow or malicious clients from blocking the accept loop.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

impl<L> axum::serve::Listener for TlsListener<L>
where
    L: axum::serve::Listener,
    L::Io: Unpin,
{
    type Addr = L::Addr;
    type Io = TlsStream<L::Io>;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, addr) = self.inner.accept().await;
            let config = self.config.load_full();
            let acceptor = tokio_rustls::TlsAcceptor::from(config);
            match time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
                Ok(Ok(tls_stream)) => return (tls_stream, addr),
                Ok(Err(e)) => {
                    debug!("TLS handshake failed: {e}");
                }
                Err(_) => {
                    debug!("TLS handshake timed out");
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

/// Produce a stream of TLS-wrapped incoming TCP connections for tonic.
///
/// Connections whose handshake fails or times out are dropped silently. The stream yields
/// `io::Result<TlsStream<TcpStream>>` and never terminates on its own, so callers should
/// drive it with a shutdown signal.
pub fn tls_incoming(
    tcp: TcpListener,
    config: Arc<ArcSwap<ServerConfig>>,
) -> impl Stream<Item = io::Result<TlsStream<TcpStream>>> + Send {
    IncomingTls {
        tcp,
        config,
        pending: None,
        sleep: None,
    }
}

type HandshakeFut =
    Pin<Box<dyn Future<Output = (io::Result<TlsStream<TcpStream>>, SocketAddr)> + Send + 'static>>;

struct IncomingTls {
    tcp: TcpListener,
    config: Arc<ArcSwap<ServerConfig>>,
    pending: Option<HandshakeFut>,
    sleep: Option<Pin<Box<tokio::time::Sleep>>>,
}

impl Stream for IncomingTls {
    type Item = io::Result<TlsStream<TcpStream>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // If we previously hit a transient accept error, wait it out before trying again.
            if let Some(sleep) = self.sleep.as_mut() {
                if sleep.as_mut().poll(cx).is_pending() {
                    return Poll::Pending;
                }
                self.sleep = None;
            }

            if let Some(fut) = self.pending.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready((Ok(stream), _addr)) => {
                        self.pending = None;
                        return Poll::Ready(Some(Ok(stream)));
                    }
                    Poll::Ready((Err(e), addr)) => {
                        debug!(%addr, "gRPC TLS handshake failed: {e}");
                        self.pending = None;
                        continue;
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }

            match self.tcp.poll_accept(cx) {
                Poll::Ready(Ok((stream, addr))) => {
                    let config = self.config.load_full();
                    let acceptor = tokio_rustls::TlsAcceptor::from(config);
                    let fut: HandshakeFut = Box::pin(async move {
                        let result = match time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(stream))
                            .await
                        {
                            Ok(inner) => inner,
                            Err(_) => Err(io::Error::new(io::ErrorKind::TimedOut, "TLS handshake")),
                        };
                        (result, addr)
                    });
                    self.pending = Some(fut);
                    continue;
                }
                Poll::Ready(Err(e)) => {
                    tracing::error!("gRPC TCP accept error: {e}");
                    self.sleep = Some(Box::pin(time::sleep(Duration::from_secs(1))));
                    continue;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Server config construction
// ---------------------------------------------------------------------------

/// Build a [`rustls::ServerConfig`] from the user-provided [`TlsConfig`].
pub fn build_server_config(tls: &TlsConfig) -> Result<ServerConfig> {
    let certs = load_certs(&tls.certfile).context("failed to load SSL certificate file")?;
    let key = load_private_key(&tls.keyfile).context("failed to load SSL key file")?;

    let config = if tls.cert_reqs > 0 {
        let ca_path = tls
            .ca_certs
            .as_deref()
            .context("--ssl-cert-reqs > 0 requires --ssl-ca-certs to be set")?;
        let ca_certs = load_certs(ca_path).context("failed to load CA certificates file")?;
        let mut root_store = rustls::RootCertStore::empty();
        for cert in ca_certs {
            root_store.add(cert).context("invalid CA certificate")?;
        }

        let verifier_builder = rustls::server::WebPkiClientVerifier::builder(Arc::new(root_store));
        let verifier = match tls.cert_reqs {
            1 => verifier_builder
                .allow_unauthenticated()
                .build()
                .context("failed to build optional client cert verifier")?,
            2 => verifier_builder
                .build()
                .context("failed to build required client cert verifier")?,
            other => bail!("invalid --ssl-cert-reqs value: {other} (expected 0, 1, or 2)"),
        };

        ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(certs, key)
            .context("failed to configure TLS certificate chain")?
    } else {
        ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .context("failed to configure TLS certificate chain")?
    };

    Ok(config)
}

fn load_certs(path: &str) -> Result<Vec<CertificateDer<'static>>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open certificate file {path}"))?;
    let mut reader = io::BufReader::new(file);
    rustls_pemfile::certs(&mut reader)
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to parse certificates from {path}"))
}

fn load_private_key(path: &str) -> Result<PrivateKeyDer<'static>> {
    let file = std::fs::File::open(path).with_context(|| format!("cannot open key file {path}"))?;
    let mut reader = io::BufReader::new(file);
    rustls_pemfile::private_key(&mut reader)?
        .with_context(|| format!("no private key found in {path}"))
}

// ---------------------------------------------------------------------------
// Certificate refresh
// ---------------------------------------------------------------------------

/// Poll interval for checking certificate file changes.
const CERT_REFRESH_INTERVAL: Duration = Duration::from_secs(5);

/// Spawn a background task that periodically checks whether the TLS certificate files have
/// changed on disk. When a change is detected the [`ServerConfig`] is rebuilt and hot-swapped
/// into the [`TlsListener`] so that new connections immediately use the updated certificates.
///
/// Equivalent to Python vLLM's `SSLCertRefresher`.
pub fn spawn_cert_refresh(
    config_holder: Arc<ArcSwap<ServerConfig>>,
    tls_config: TlsConfig,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut last_mtimes = collect_file_mtimes(&tls_config);
        info!(
            keyfile = %tls_config.keyfile,
            certfile = %tls_config.certfile,
            ca_certs = ?tls_config.ca_certs,
            "watching TLS certificate files for changes"
        );

        loop {
            time::sleep(CERT_REFRESH_INTERVAL).await;
            let current_mtimes = collect_file_mtimes(&tls_config);
            if current_mtimes == last_mtimes {
                continue;
            }
            info!("TLS certificate file change detected, reloading...");
            match build_server_config(&tls_config) {
                Ok(new_config) => {
                    config_holder.store(Arc::new(new_config));
                    last_mtimes = current_mtimes;
                    info!("TLS certificates reloaded successfully");
                }
                Err(e) => {
                    warn!("failed to reload TLS certificates, keeping previous config: {e:#}");
                }
            }
        }
    })
}

/// Collect modification times for all configured TLS certificate files.
fn collect_file_mtimes(tls: &TlsConfig) -> Vec<Option<SystemTime>> {
    let mut paths: Vec<&str> = vec![&tls.keyfile, &tls.certfile];
    if let Some(ref ca) = tls.ca_certs {
        paths.push(ca);
    }
    paths
        .iter()
        .map(|p| std::fs::metadata(p).ok().and_then(|m| m.modified().ok()))
        .collect()
}
