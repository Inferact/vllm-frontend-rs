use std::sync::Arc;

use thiserror_ext::AsReport;
use tokio::sync::mpsc;
use tracing::{debug, warn};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{XSubSocket, ZmqMessage};

use crate::client::imp::ClientInner;
use crate::coordinator::handle::{CoordinatorCommand, CoordinatorState};
use crate::error::{Error, Result, bail_unexpected_coordinator_output};
use crate::protocol::{OpaqueValue, decode_msgpack, encode_msgpack};

/// Background half of an external Python-owned coordinator connection.
///
/// This owns the command receiver and one frontend-facing XSUB socket. It mirrors
/// the subset of Python's coordinator protocol needed by the Rust bootstrapped
/// frontend: receive `(counts, wave, running)` publishes, ignore `counts`, and
/// send `(exclude_engine_index, wave)` wakeup messages when the first request
/// arrives while engines are paused.
pub(crate) struct ExternalCoordinatorService {
    state: Arc<CoordinatorState>,
    command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
    socket: XSubSocket,
}

impl ExternalCoordinatorService {
    pub(super) fn new(
        state: Arc<CoordinatorState>,
        command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
        socket: XSubSocket,
    ) -> Self {
        Self {
            state,
            command_rx,
            socket,
        }
    }

    /// Apply one frontend-originated command to th
    async fn handle_command(&mut self, command: CoordinatorCommand) -> Result<()> {
        match command {
            CoordinatorCommand::FirstRequest {
                target_engine_id,
                wave,
            } => {
                let target_engine_index = target_engine_id.engine_index().ok_or_else(|| {
                    Error::UnsupportedCoordinatorEngineId {
                        engine_id: target_engine_id.to_vec(),
                    }
                })?;
                debug!(
                    wave,
                    exclude_engine_index = target_engine_index,
                    "notifying external coordinator about first request while engines were paused"
                );
                let payload = encode_msgpack(&(target_engine_index, wave))?;
                self.socket.send(ZmqMessage::from(payload)).await?;
            }
        }
        Ok(())
    }

    /// Apply one publish received from the xsub socket containing a coordinator state update.
    async fn handle_publish(&mut self, message: ZmqMessage) -> Result<()> {
        let frames = message.into_vec();
        if frames.len() != 1 {
            bail_unexpected_coordinator_output!(
                "received malformed external coordinator publish with {} frame(s)",
                frames.len()
            );
        }

        // Note: we ignore `counts` since the client doesn't track stats for routing decisions.
        let (_counts, wave, running): (OpaqueValue, u32, bool) = decode_msgpack(&frames[0])?;
        let mut state = self.state.lock();
        state.current_wave = wave;
        state.engines_running = running;
        Ok(())
    }

    /// Drive the coordinator event loop until either side of the control plane
    /// is closed or a fatal error is observed.
    pub(crate) async fn run(mut self, inner: Arc<ClientInner>) {
        let Err(error) = try {
            loop {
                tokio::select! {
                    // Received frontend-originated command from the handle.
                    command = self.command_rx.recv() => {
                        let Some(command) = command else {
                            warn!("external coordinator command channel closed, shutting down service");
                            return;
                        };
                        self.handle_command(command).await?;
                    }
                    // Received publish from the external coordinator socket.
                    publish = self.socket.recv() => {
                        let publish = publish.map_err(Error::from)?;
                        self.handle_publish(publish).await?;
                    }
                }
            }
        };

        warn!(
            error = %error.as_report(),
            "external coordinator service exiting with error"
        );
        inner.close_registries(Arc::new(error));
    }
}
