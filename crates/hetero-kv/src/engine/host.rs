//! Host reference engine: synchronous `memmove` between host regions.

use super::{Capabilities, Completion, CopyEngine, CopyJob, Representation};
use crate::error::Result;
use crate::memory::MemoryKind;

/// Copies between pageable/pinned host regions of this node on the calling thread (identity
/// routes only). Every job is complete when `submit` returns, so `quiesce` is trivially
/// satisfied. It serves only host kinds: a host engine is never a silent stand-in for a device
/// route.
pub struct HostEngine {
    caps: Capabilities,
}

impl Default for HostEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HostEngine {
    /// Engine serving every host-to-host route and representation.
    pub fn new() -> Self {
        let mut caps = Capabilities::new();
        for src in [MemoryKind::Host, MemoryKind::HostPinned] {
            for dst in [MemoryKind::Host, MemoryKind::HostPinned] {
                caps = caps.allow(src, dst, &Representation::ALL);
            }
        }
        HostEngine { caps }
    }
}

struct Ready(u64);

impl Completion for Ready {
    fn poll(&mut self) -> Result<Option<u64>> {
        Ok(Some(self.0))
    }
    fn wait(&mut self) -> Result<u64> {
        Ok(self.0)
    }
}

impl CopyEngine for HostEngine {
    fn name(&self) -> &str {
        "host"
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn submit(&self, job: &CopyJob<'_>) -> Result<Box<dyn Completion>> {
        let mut copied = 0;
        for s in job.spans {
            let (src, dst) = (&job.src[s.src as usize], &job.dst[s.dst as usize]);
            debug_assert!(job.representation == job.dst_representation);
            // SAFETY: `Route::submit` checked locations (local host), bounds and destination
            // overlap, and claimed the ranges, so neither owners nor other jobs touch them.
            // `copy` tolerates a source and destination in the same region.
            unsafe {
                std::ptr::copy(
                    (src.addr() as *const u8).add(s.src_offset as usize),
                    (dst.addr() as *mut u8).add(s.dst_offset as usize),
                    s.len as usize,
                );
            }
            copied += s.len;
        }
        Ok(Box::new(Ready(copied)))
    }

    fn quiesce(&self) -> Result<()> {
        Ok(())
    }
}
