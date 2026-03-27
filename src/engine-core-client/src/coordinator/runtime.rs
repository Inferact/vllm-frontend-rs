use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use tracing::warn;
use zeromq::prelude::SocketSend;
use zeromq::{XPubSocket, ZmqMessage};

use crate::client::imp::ClientInner;
use crate::error::{Error, Result};
use crate::protocol::{
    ClassifiedEngineCoreOutputs, DpControlMessage, EngineCoreOutputs, EngineCoreRequestType,
    OtherEngineCoreOutputs, encode_msgpack,
};
use crate::transport::EngineId;

/// Snapshot to the coordinator state for request routing and stamping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoordinatorStateSnapshot {
    /// The current DP wave, which will be stamped on outgoing requests.
    pub current_wave: u32,
    /// Whether the engines are currently running or paused, which determines if the frontend
    /// must trigger a new wave on the next request.
    pub engines_running: bool,
}

/// Shared in-process coordinator state.
type CoordinatorState = Mutex<CoordinatorStateSnapshot>;

/// Commands sent from the frontend request path into the background runner.
///
/// At this stage the only frontend-initiated transition is "the first request
/// arrived while all engines were paused".
#[derive(Debug)]
enum CoordinatorCommand {
    FirstRequest {
        target_engine_id: EngineId,
        wave: u32,
    },
}

/// Frontend-facing coordinator handle used by `EngineCoreClient::call()`.
///
/// This side stays intentionally small: it can read the latest wave snapshot and
/// enqueue a `FirstRequest` transition when the request path observes the system
/// in the paused state.
#[derive(Clone)]
pub(crate) struct CoordinatorHandle {
    state: Arc<CoordinatorState>,
    command_tx: mpsc::UnboundedSender<CoordinatorCommand>,
}

impl CoordinatorHandle {
    /// Build the paired frontend handle and background runner around one
    /// engine-facing coordinator broadcast socket.
    pub(crate) fn new(coordinator_input: XPubSocket) -> (Self, CoordinatorRunner) {
        let state = Arc::new(Mutex::new(CoordinatorStateSnapshot {
            current_wave: 0,
            engines_running: false,
        }));
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        (
            Self {
                state: state.clone(),
                command_tx,
            },
            CoordinatorRunner::new(state, command_rx, coordinator_input),
        )
    }

    /// Snapshot the coordinator state for request routing and stamping.
    pub(crate) fn snapshot(&self) -> CoordinatorStateSnapshot {
        *self.state.lock()
    }

    /// Notify the runner that a new request arrived while engines were paused.
    ///
    /// The handle flips `engines_running` optimistically so concurrent request
    /// submissions coalesce behind one `START_DP_WAVE` broadcast instead of all
    /// trying to trigger the wave independently.
    pub(crate) fn notify_first_request(&self, target_engine_id: EngineId) -> Result<()> {
        let wave = {
            let mut state = self.state.lock();
            if state.engines_running {
                return Ok(());
            }
            state.engines_running = true;
            state.current_wave
        };

        let command = CoordinatorCommand::FirstRequest {
            target_engine_id,
            wave,
        };
        if self.command_tx.send(command).is_err() {
            self.state.lock().engines_running = false;
            return Err(Error::ControlClosed(
                "in-process coordinator command channel already shut down".to_string(),
            ));
        }

        Ok(())
    }
}

/// Background half of the in-process coordinator.
///
/// This owns the command receiver and the engine-facing coordinator input socket.
/// It is the single place where wave transitions are serialized and where
/// `START_DP_WAVE` broadcasts are emitted.
pub(crate) struct CoordinatorRunner {
    state: Arc<CoordinatorState>,
    command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
    coordinator_input: XPubSocket,
}

impl CoordinatorRunner {
    fn new(
        state: Arc<CoordinatorState>,
        command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
        coordinator_input: XPubSocket,
    ) -> Self {
        Self {
            state,
            command_rx,
            coordinator_input,
        }
    }

    /// Broadcast Python-compatible `START_DP_WAVE` to all connected engines.
    async fn broadcast_start_wave(&mut self, wave: u32, exclude_engine_index: u32) -> Result<()> {
        let payload = encode_msgpack(&(wave, exclude_engine_index))?;
        self.coordinator_input
            .send(
                ZmqMessage::try_from(vec![
                    EngineCoreRequestType::StartDpWave.as_frame(),
                    payload.into(),
                ])
                .expect("coordinator START_DP_WAVE message must contain two frames"),
            )
            .await?;
        Ok(())
    }

    /// Apply one frontend-originated command to the coordinator state machine.
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
                self.state.lock().current_wave = wave;
                self.broadcast_start_wave(wave, target_engine_index).await?;
            }
        }
        Ok(())
    }

    /// Apply one engine-originated control output to the coordinator state machine.
    ///
    /// `WaveComplete` advances the wave and returns the group to the paused
    /// state. `StartWave` requests a rebroadcast, which mirrors Python's
    /// behavior when an engine needs to re-open the current or a newer wave.
    async fn handle_outputs(&mut self, outputs: EngineCoreOutputs) -> Result<()> {
        match outputs.classify() {
            ClassifiedEngineCoreOutputs::Other(OtherEngineCoreOutputs::DpControl {
                engine_index,
                control,
                ..
            }) => match control {
                DpControlMessage::WaveComplete(wave) => {
                    let mut state = self.state.lock();
                    if wave >= state.current_wave {
                        state.current_wave = wave + 1;
                        state.engines_running = false;
                    }
                }
                DpControlMessage::StartWave(wave) => {
                    let should_broadcast = {
                        let mut state = self.state.lock();
                        if wave > state.current_wave
                            || (wave == state.current_wave && !state.engines_running)
                        {
                            state.current_wave = wave;
                            state.engines_running = true;
                            true
                        } else {
                            false
                        }
                    };
                    if should_broadcast {
                        self.broadcast_start_wave(wave, engine_index).await?;
                    }
                }
            },
            ClassifiedEngineCoreOutputs::RequestBatch(batch) => {
                if batch.scheduler_stats.is_some() {
                    warn!(
                        engine_index = batch.engine_index,
                        "ignoring scheduler_stats on in-process coordinator control path"
                    );
                } else if !batch.outputs.is_empty() || batch.finished_requests.is_some() {
                    warn!(outputs = ?batch, "ignoring request outputs on in-process coordinator control path");
                }
            }
            ClassifiedEngineCoreOutputs::Utility(output) => {
                warn!(outputs = ?output, "ignoring utility output on in-process coordinator control path");
            }
            ClassifiedEngineCoreOutputs::Other(OtherEngineCoreOutputs::Raw(raw)) => {
                warn!(outputs = ?raw, "ignoring raw engine-core output on in-process coordinator control path");
            }
        }

        Ok(())
    }

    /// Drive the coordinator event loop until either side of the control plane
    /// is closed or a fatal error is observed.
    ///
    /// Any fatal error closes the main client registries so request streams and
    /// future calls observe a stable shutdown cause.
    pub(crate) async fn run(
        mut self,
        mut output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
        inner: Arc<ClientInner>,
    ) {
        loop {
            tokio::select! {
                command = self.command_rx.recv() => {
                    let Some(command) = command else {
                        return;
                    };
                    if let Err(error) = self.handle_command(command).await {
                        inner.close_registries(Arc::new(error));
                        return;
                    }
                }
                outputs = output_rx.recv() => {
                    let Some(outputs) = outputs else {
                        return;
                    };
                    let outputs = match outputs {
                        Ok(outputs) => outputs,
                        Err(error) => {
                            inner.close_registries(Arc::new(error));
                            return;
                        }
                    };

                    if let Err(error) = self.handle_outputs(outputs).await {
                        inner.close_registries(Arc::new(error));
                        return;
                    }
                }
            }
        }
    }
}
