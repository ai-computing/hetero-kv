//! L2 engines: the [`CopyEngine`] trait, the capability matrix, explicit route selection, and
//! transfer handles that drive the L1 ownership state machine.
//!
//! An engine copies byte spans between regions. It never chooses a route: callers select a
//! [`Route`] for `(source kind, destination kind, representation)` and selection fails with
//! [`Error::Unsupported`] if the engine does not advertise that triple. There is no fallback.

mod host;

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use host::HostEngine;

use crate::error::{invalid, Error, Result};
use crate::memory::{MemoryKind, Phase, Quiesced, RegionRef};

/// Physical element encoding of KV data. Distinct values never alias: BF16, FP16 and
/// DLFloat16 are all 2 bytes but are not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    /// bfloat16.
    Bf16,
    /// IEEE half precision.
    Fp16,
    /// FuriosaAI DLFloat16.
    Dlfloat16,
    /// IEEE single precision (CPU reference engines and tests use it).
    Fp32,
}

impl Representation {
    /// All representations.
    pub const ALL: [Representation; 4] = [
        Representation::Bf16,
        Representation::Fp16,
        Representation::Dlfloat16,
        Representation::Fp32,
    ];

    /// Bytes per element.
    pub fn element_bytes(self) -> u64 {
        match self {
            Representation::Fp32 => 4,
            _ => 2,
        }
    }

    /// Canonical name, also used on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Representation::Bf16 => "bf16",
            Representation::Fp16 => "fp16",
            Representation::Dlfloat16 => "dlfloat16",
            Representation::Fp32 => "fp32",
        }
    }
}

impl fmt::Display for Representation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Representation {
    type Err = Error;
    /// Accepts canonical names and common aliases (`bfloat16`, `torch.float16`, ...).
    fn from_str(s: &str) -> Result<Self> {
        let s = s.strip_prefix("torch.").unwrap_or(s);
        Ok(match s {
            "bf16" | "bfloat16" => Representation::Bf16,
            "fp16" | "float16" | "half" => Representation::Fp16,
            "dlfloat16" | "dlf16" => Representation::Dlfloat16,
            "fp32" | "float32" | "float" => Representation::Fp32,
            _ => return invalid(format!("unknown KV representation {s:?}")),
        })
    }
}

/// Who initiates a copy. Recorded for routing and reported to engines; P0 engines copy the
/// same way in both directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Push: the initiator owns the source and writes into a (possibly remote) destination.
    Write,
    /// Pull: the initiator owns the destination and reads a (possibly remote) source.
    Read,
}

impl Direction {
    /// Lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Direction::Write => "write",
            Direction::Read => "read",
        }
    }
}

impl FromStr for Direction {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "write" | "push" => Ok(Direction::Write),
            "read" | "pull" => Ok(Direction::Read),
            _ => invalid(format!("unknown direction {s:?}")),
        }
    }
}

/// One contiguous copy: `len` bytes from `src[src]` at `src_offset` to `dst[dst]` at
/// `dst_offset`, where `src`/`dst` index the region lists of a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// Index into the job's source regions.
    pub src: u32,
    /// Byte offset in that source region.
    pub src_offset: u64,
    /// Index into the job's destination regions.
    pub dst: u32,
    /// Byte offset in that destination region.
    pub dst_offset: u64,
    /// Length in bytes (positive).
    pub len: u64,
}

/// What an engine is asked to do. Spans were validated against the regions before submission.
pub struct CopyJob<'a> {
    /// Source regions.
    pub src: &'a [RegionRef],
    /// Destination regions.
    pub dst: &'a [RegionRef],
    /// Byte spans; destinations never overlap.
    pub spans: &'a [Span],
    /// Element encoding of the bytes (engines may refuse or convert per their capabilities).
    pub representation: Representation,
    /// Push or pull.
    pub direction: Direction,
}

impl CopyJob<'_> {
    /// Total bytes the job moves.
    pub fn bytes(&self) -> u64 {
        self.spans.iter().map(|s| s.len).sum()
    }
}

/// Completion of one submitted job.
pub trait Completion: Send {
    /// Non-blocking check. `Ok(Some(bytes))` once complete.
    fn poll(&mut self) -> Result<Option<u64>>;
    /// Block until complete; returns the number of bytes the engine verified as copied.
    fn wait(&mut self) -> Result<u64>;
}

/// A copy engine. Implementations: [`HostEngine`]; Python objects (via `hetero-kv-py`);
/// CUDA/RNGD/NIXL in later phases.
pub trait CopyEngine: Send + Sync {
    /// Short name for receipts and errors.
    fn name(&self) -> &str;
    /// Routes this engine serves.
    fn capabilities(&self) -> &Capabilities;
    /// Start copying. Must not block on device completion. An error means the engine may or
    /// may not have started work; the regions become `Uncertain`.
    fn submit(&self, job: &CopyJob<'_>) -> Result<Box<dyn Completion>>;
    /// Wait until no accepted job is running (or fail if that cannot be established).
    fn quiesce(&self) -> Result<()>;
}

/// Capability matrix: the `(source kind, destination kind, representation)` triples served.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Capabilities {
    routes: Vec<(MemoryKind, MemoryKind, Representation)>,
}

impl Capabilities {
    /// Empty matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple for each listed representation.
    pub fn allow(
        mut self,
        src: MemoryKind,
        dst: MemoryKind,
        representations: &[Representation],
    ) -> Self {
        for &r in representations {
            if !self.supports(src, dst, r) {
                self.routes.push((src, dst, r));
            }
        }
        self
    }

    /// True if the triple is served.
    pub fn supports(&self, src: MemoryKind, dst: MemoryKind, repr: Representation) -> bool {
        self.routes.contains(&(src, dst, repr))
    }

    /// All served triples.
    pub fn routes(&self) -> &[(MemoryKind, MemoryKind, Representation)] {
        &self.routes
    }
}

/// An explicitly selected engine route.
#[derive(Clone)]
pub struct Route {
    engine: Arc<dyn CopyEngine>,
    src: MemoryKind,
    dst: MemoryKind,
    repr: Representation,
}

impl fmt::Debug for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Route({}: {} -> {}, {})",
            self.engine.name(),
            self.src,
            self.dst,
            self.repr
        )
    }
}

impl Route {
    /// Select a route. Fails with [`Error::Unsupported`] if the engine does not serve it.
    pub fn select(
        engine: Arc<dyn CopyEngine>,
        src: MemoryKind,
        dst: MemoryKind,
        repr: Representation,
    ) -> Result<Route> {
        if !engine.capabilities().supports(src, dst, repr) {
            return Err(Error::Unsupported(format!(
                "engine {} has no route {src} -> {dst} for {repr}",
                engine.name()
            )));
        }
        Ok(Route {
            engine,
            src,
            dst,
            repr,
        })
    }

    /// Source memory kind.
    pub fn src(&self) -> MemoryKind {
        self.src
    }
    /// Destination memory kind.
    pub fn dst(&self) -> MemoryKind {
        self.dst
    }
    /// Representation.
    pub fn representation(&self) -> Representation {
        self.repr
    }
    /// Engine name.
    pub fn engine_name(&self) -> &str {
        self.engine.name()
    }

    /// Validate and submit a job. Rejections (bad spans, wrong kinds, busy or uncertain regions)
    /// leave every region unchanged. Once handed to the engine, the regions are `InFlight`.
    pub fn submit(
        &self,
        src: &[RegionRef],
        dst: &[RegionRef],
        spans: &[Span],
        direction: Direction,
    ) -> Result<Transfer> {
        self.validate(src, dst, spans)?;
        // Mark every distinct region a span touches InFlight, rolling back if one is
        // unavailable. Untouched regions of the same buffers stay free for other transfers
        // (e.g. other layers of a layer-streamed request).
        let mut regions: Vec<RegionRef> = Vec::new();
        for s in spans {
            for r in [&src[s.src as usize], &dst[s.dst as usize]] {
                if !regions.iter().any(|x| x.same(r)) {
                    regions.push(r.clone());
                }
            }
        }
        let mut started: Vec<(usize, Phase)> = Vec::new();
        for (i, r) in regions.iter().enumerate() {
            match r.begin() {
                Ok(prev) => started.push((i, prev)),
                Err(e) => {
                    for (j, prev) in started {
                        regions[j].rollback(prev);
                    }
                    return Err(e);
                }
            }
        }
        let job = CopyJob {
            src,
            dst,
            spans,
            representation: self.repr,
            direction,
        };
        let bytes = job.bytes();
        // From here on the transfer owns the regions' fate: drop or failure means Uncertain.
        let mut transfer = Transfer {
            completion: None,
            regions,
            bytes,
            spans: spans.len(),
            outcome: Outcome::Pending,
        };
        match self.engine.submit(&job) {
            Ok(c) => {
                transfer.completion = Some(c);
                Ok(transfer)
            }
            Err(e) => {
                let e = transfer.fail(format!("{} submit failed: {e}", self.engine.name()));
                Err(e)
            }
        }
    }

    fn validate(&self, src: &[RegionRef], dst: &[RegionRef], spans: &[Span]) -> Result<()> {
        if spans.is_empty() {
            return invalid("empty copy plan");
        }
        for (side, regions, kind) in [("source", src, self.src), ("destination", dst, self.dst)] {
            if let Some(r) = regions.iter().find(|r| r.kind() != kind) {
                return Err(Error::Unsupported(format!(
                    "{side} region is {} but the route expects {kind}",
                    r.kind()
                )));
            }
        }
        let mut writes: Vec<(usize, u64, u64)> = Vec::with_capacity(spans.len());
        for s in spans {
            let (Some(a), Some(b)) = (src.get(s.src as usize), dst.get(s.dst as usize)) else {
                return invalid("span names an unknown region");
            };
            let in_bounds =
                |off: u64, len: usize| off.checked_add(s.len).is_some_and(|end| end <= len as u64);
            if s.len == 0 || !in_bounds(s.src_offset, a.len()) || !in_bounds(s.dst_offset, b.len())
            {
                return invalid("span is empty or outside its region");
            }
            writes.push((s.dst as usize, s.dst_offset, s.dst_offset + s.len));
        }
        // Destinations must not overlap, including two indices naming the same region.
        let key = |i: usize| dst.iter().position(|r| r.same(&dst[i])).unwrap_or(i);
        let mut writes: Vec<_> = writes.into_iter().map(|(i, a, b)| (key(i), a, b)).collect();
        writes.sort_unstable();
        if writes
            .windows(2)
            .any(|w| w[0].0 == w[1].0 && w[1].1 < w[0].2)
        {
            return invalid("overlapping destination spans");
        }
        Ok(())
    }

    /// Quiesce the engine and return the proof needed to settle uncertain regions.
    pub fn quiesce(&self) -> Result<Quiesced> {
        self.engine.quiesce()?;
        Ok(Quiesced::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Pending,
    Done,
    Failed(String),
}

/// Handle to a submitted job. Regions stay `InFlight` until it completes; dropping it before
/// completion marks them `Uncertain` (the engine may still be copying).
pub struct Transfer {
    completion: Option<Box<dyn Completion>>,
    regions: Vec<RegionRef>,
    bytes: u64,
    spans: usize,
    outcome: Outcome,
}

impl fmt::Debug for Transfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transfer({} bytes, {:?})", self.bytes, self.outcome)
    }
}

impl Transfer {
    /// Planned bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Number of spans.
    pub fn spans(&self) -> usize {
        self.spans
    }

    /// True once completed successfully.
    pub fn is_done(&self) -> bool {
        self.outcome == Outcome::Done
    }

    /// Non-blocking completion check.
    pub fn poll(&mut self) -> Result<bool> {
        match &self.outcome {
            Outcome::Done => return Ok(true),
            Outcome::Failed(m) => return Err(Error::Transfer(m.clone())),
            Outcome::Pending => {}
        }
        let polled = self.completion.as_mut().map(|c| c.poll());
        match polled {
            Some(Ok(None)) => Ok(false),
            Some(Ok(Some(n))) => self.complete(n).map(|_| true),
            Some(Err(e)) => Err(self.fail(e.to_string())),
            None => Err(self.fail("transfer has no completion".into())),
        }
    }

    /// Block until completion and verify the engine copied exactly the planned bytes.
    pub fn wait(&mut self) -> Result<u64> {
        match &self.outcome {
            Outcome::Done => return Ok(self.bytes),
            Outcome::Failed(m) => return Err(Error::Transfer(m.clone())),
            Outcome::Pending => {}
        }
        let waited = self.completion.as_mut().map(|c| c.wait());
        match waited {
            Some(Ok(n)) => self.complete(n),
            Some(Err(e)) => Err(self.fail(e.to_string())),
            None => Err(self.fail("transfer has no completion".into())),
        }
    }

    fn complete(&mut self, copied: u64) -> Result<u64> {
        if copied != self.bytes {
            return Err(self.fail(format!(
                "inconsistent completion: engine reported {copied} of {} bytes",
                self.bytes
            )));
        }
        self.completion = None;
        self.outcome = Outcome::Done;
        for r in &self.regions {
            r.end(true);
        }
        Ok(self.bytes)
    }

    fn fail(&mut self, msg: String) -> Error {
        self.completion = None;
        self.outcome = Outcome::Failed(msg.clone());
        for r in &self.regions {
            r.end(false);
        }
        Error::Transfer(msg)
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        if self.outcome == Outcome::Pending {
            self.fail("transfer handle dropped before completion".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Lent, Region};
    use std::sync::Mutex;

    fn host_pair(n: usize) -> (Lent, Lent) {
        let mut a = Region::host(n).unwrap();
        for (i, b) in a.as_mut_slice().unwrap().iter_mut().enumerate() {
            *b = i as u8;
        }
        (a.lend(), Region::host(n).unwrap().lend())
    }

    fn host_route() -> Route {
        Route::select(
            Arc::new(HostEngine::new()),
            MemoryKind::Host,
            MemoryKind::Host,
            Representation::Bf16,
        )
        .unwrap()
    }

    #[test]
    fn capability_rejection_has_no_fallback() {
        let e = Route::select(
            Arc::new(HostEngine::new()),
            MemoryKind::Host,
            MemoryKind::Cuda,
            Representation::Bf16,
        )
        .unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)));
        assert!(Capabilities::new()
            .allow(MemoryKind::Host, MemoryKind::Host, &[Representation::Fp16])
            .supports(MemoryKind::Host, MemoryKind::Host, Representation::Fp16));
    }

    #[test]
    fn host_copy_moves_bytes_and_settles_done() {
        let (a, b) = host_pair(64);
        let route = host_route();
        let span = Span {
            src: 0,
            src_offset: 8,
            dst: 0,
            dst_offset: 0,
            len: 16,
        };
        let mut t = route
            .submit(&[a.share()], &[b.share()], &[span], Direction::Write)
            .unwrap();
        assert_eq!(b.phase(), Phase::InFlight);
        assert_eq!(t.wait().unwrap(), 16);
        drop(t);
        assert_eq!(b.phase(), Phase::Done);
        let b = b.reclaim().unwrap();
        assert_eq!(
            &b.as_slice().unwrap()[..16],
            &(8u8..24).collect::<Vec<_>>()[..]
        );
    }

    #[test]
    fn rejected_submissions_change_nothing() {
        let (a, b) = host_pair(64);
        let route = host_route();
        let ok = Span {
            src: 0,
            src_offset: 0,
            dst: 0,
            dst_offset: 0,
            len: 16,
        };
        let cases = [
            vec![],
            vec![Span { len: 0, ..ok }],
            vec![Span {
                dst_offset: 60,
                ..ok
            }],
            vec![Span { dst: 1, ..ok }],
            vec![
                ok,
                Span {
                    src_offset: 16,
                    dst_offset: 8,
                    ..ok
                },
            ],
        ];
        for spans in cases {
            let e = route
                .submit(&[a.share()], &[b.share()], &spans, Direction::Write)
                .unwrap_err();
            assert!(e.is_rejection(), "{e}");
            assert_eq!((a.phase(), b.phase()), (Phase::Lent, Phase::Lent));
        }
        // Same region twice in the destination list still counts as overlap.
        let e = route
            .submit(
                &[a.share()],
                &[b.share(), b.share()],
                &[ok, Span { dst: 1, ..ok }],
                Direction::Write,
            )
            .unwrap_err();
        assert!(e.is_rejection());
        // Kind mismatch against the route.
        let cuda = unsafe { Region::external(MemoryKind::Cuda, 1, 64, None).unwrap() }.lend();
        let e = route
            .submit(&[a.share()], &[cuda.share()], &[ok], Direction::Write)
            .unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)));
    }

    /// An engine whose completions are controlled by the test.
    struct Scripted {
        caps: Capabilities,
        result: Mutex<Option<Result<u64>>>,
        quiesce_ok: bool,
    }
    struct ScriptedCompletion(Option<Result<u64>>);
    impl Completion for ScriptedCompletion {
        fn poll(&mut self) -> Result<Option<u64>> {
            Ok(None)
        }
        fn wait(&mut self) -> Result<u64> {
            self.0.take().unwrap_or(Ok(0))
        }
    }
    impl CopyEngine for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> &Capabilities {
            &self.caps
        }
        fn submit(&self, _job: &CopyJob<'_>) -> Result<Box<dyn Completion>> {
            match self.result.lock().unwrap().take() {
                Some(Err(e)) if e.is_rejection() => Err(Error::Transfer(e.to_string())),
                r => Ok(Box::new(ScriptedCompletion(r))),
            }
        }
        fn quiesce(&self) -> Result<()> {
            if self.quiesce_ok {
                Ok(())
            } else {
                Err(Error::Transfer("device unresponsive".into()))
            }
        }
    }

    fn scripted(result: Result<u64>) -> Route {
        let caps =
            Capabilities::new().allow(MemoryKind::Host, MemoryKind::Host, &Representation::ALL);
        Route::select(
            Arc::new(Scripted {
                caps,
                result: Mutex::new(Some(result)),
                quiesce_ok: true,
            }),
            MemoryKind::Host,
            MemoryKind::Host,
            Representation::Fp16,
        )
        .unwrap()
    }

    #[test]
    fn short_completion_and_dropped_handles_are_uncertain() {
        let span = Span {
            src: 0,
            src_offset: 0,
            dst: 0,
            dst_offset: 0,
            len: 32,
        };
        // Engine claims fewer bytes than planned: uncertain, not done.
        let (a, b) = host_pair(64);
        let mut t = scripted(Ok(31))
            .submit(&[a.share()], &[b.share()], &[span], Direction::Read)
            .unwrap();
        assert!(matches!(t.wait(), Err(Error::Transfer(_))));
        assert!(t.wait().is_err(), "failure is sticky");
        drop(t);
        assert_eq!(b.phase(), Phase::Uncertain);
        // Handle dropped before completion: uncertain.
        let (a, b) = host_pair(64);
        let route = scripted(Ok(32));
        let t = route
            .submit(&[a.share()], &[b.share()], &[span], Direction::Write)
            .unwrap();
        drop(t);
        assert_eq!(a.phase(), Phase::Uncertain);
        // Resubmission refused until quiesced.
        assert!(route
            .submit(&[a.share()], &[b.share()], &[span], Direction::Write)
            .is_err());
        let proof = route.quiesce().unwrap();
        assert!(a.reclaim_quiesced(&proof).is_ok());
    }

    #[test]
    fn failed_quiesce_yields_no_proof() {
        let caps =
            Capabilities::new().allow(MemoryKind::Host, MemoryKind::Host, &Representation::ALL);
        let route = Route::select(
            Arc::new(Scripted {
                caps,
                result: Mutex::new(None),
                quiesce_ok: false,
            }),
            MemoryKind::Host,
            MemoryKind::Host,
            Representation::Bf16,
        )
        .unwrap();
        assert!(route.quiesce().is_err());
    }
}
