use std::path::Path;

use tempfile::TempDir;

/// Per-test IPC endpoint namespace backed by a unique temporary directory.
///
/// Using one directory per test avoids endpoint collisions without requiring
/// ad-hoc unique-name generation at each call site.
#[derive(Debug)]
pub struct IpcNamespace {
    dir: TempDir,
}

impl IpcNamespace {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            dir: TempDir::new()?,
        })
    }

    pub fn endpoint(&self, name: impl AsRef<Path>) -> String {
        let path = self.dir.path().join(name);
        format!("ipc://{}", path.to_string_lossy())
    }

    pub fn handshake_endpoint(&self) -> String {
        self.endpoint("handshake.sock")
    }

    pub fn input_endpoint(&self) -> String {
        self.endpoint("input.sock")
    }

    pub fn output_endpoint(&self) -> String {
        self.endpoint("output.sock")
    }
}
