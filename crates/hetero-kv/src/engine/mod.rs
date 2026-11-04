//! L2 engines: the [`CopyEngine`] trait, the capability matrix, explicit route selection, and
//! transfer handles that drive the L1 ownership state machine.
//!
//! An engine copies byte spans between regions. It never chooses a route: callers select a
//! [`Route`] for `(source location, destination location, source representation, destination
//! representation)` and selection fails with [`Error::Unsupported`] if the engine does not
//! advertise it. There is no fallback. A route whose representations differ converts on the way
//! (e.g. BF16 -> DLFloat16 for RNGD -> ATOM); P0 engines declare identity routes only.
//!
//! A [`Route`] is cheap to clone and `Send + Sync`: one engine serves many sessions at once.

mod host;

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use host::HostEngine;

use crate::error::{invalid, Error, Result};
use crate::memory::{Location, MemoryKind, Quiesced, RegionRef};

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

    /// Bytes of `to` holding the elements of `len` bytes of `self`.
    pub fn convert_len(self, len: u64, to: Representation) -> u64 {
        len / self.element_bytes() * to.element_bytes()
    }

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

/// One contiguous copy: `len` source bytes from `src[src]` at `src_offset` to `dst[dst]` at
/// `dst_offset`, where `src`/`dst` index the region lists of a job. On a converting route the
/// destination range is `CopyJob::dst_len(len)` bytes long.
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
    /// Source length in bytes (positive, whole source elements).
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
    /// Element encoding of the source bytes.
    pub representation: Representation,
    /// Element encoding to write (differs only on converting routes).
    pub dst_representation: Representation,
    /// Push or pull.
    pub direction: Direction,
}

impl CopyJob<'_> {
    /// Total source bytes the job reads; completions report this number.
    pub fn bytes(&self) -> u64 {
        self.spans.iter().map(|s| s.len).sum()
    }

    /// Destination bytes written for `len` source bytes.
    pub fn dst_len(&self, len: u64) -> u64 {
        self.representation
            .convert_len(len, self.dst_representation)
    }
}

/// Completion of one submitted job.
pub trait Completion: Send {
    /// Non-blocking check. `Ok(Some(bytes))` once complete.
    fn poll(&mut self) -> Result<Option<u64>>;
    /// Block until complete; returns the number of source bytes the engine verified as copied.
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

/// Devices an engine serves: one memory kind, and either one device index or all of them.
/// Written `kind` or `kind:*` (any index) and `kind:index`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePattern {
    /// Memory kind.
    pub kind: MemoryKind,
    /// `None` matches every device index of the kind.
    pub index: Option<u32>,
}

impl DevicePattern {
    /// Every device of `kind`.
    pub const fn any(kind: MemoryKind) -> Self {
        DevicePattern { kind, index: None }
    }

    /// Device `index` of `kind`.
    pub const fn at(kind: MemoryKind, index: u32) -> Self {
        DevicePattern {
            kind,
            index: Some(index),
        }
    }

    /// True if `loc` is one of the devices (nodes are checked by `Route::select`: P0 routes are
    /// local only).
    pub fn matches(&self, loc: Location) -> bool {
        self.kind == loc.kind && self.index.is_none_or(|i| i == loc.index)
    }
}

impl From<MemoryKind> for DevicePattern {
    fn from(kind: MemoryKind) -> Self {
        DevicePattern::any(kind)
    }
}

impl fmt::Display for DevicePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.index {
            Some(i) => write!(f, "{}:{i}", self.kind),
            None => write!(f, "{}:*", self.kind),
        }
    }
}

impl FromStr for DevicePattern {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.split_once(':') {
            None => Ok(DevicePattern::any(s.parse()?)),
            Some((k, "*")) => Ok(DevicePattern::any(k.parse()?)),
            Some((k, i)) => Ok(DevicePattern::at(
                k.parse()?,
                i.parse()
                    .map_err(|_| Error::Invalid(format!("bad device pattern {s:?}")))?,
            )),
        }
    }
}

/// One served route: device patterns and the source -> destination representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteCap {
    /// Source devices.
    pub src: DevicePattern,
    /// Destination devices.
    pub dst: DevicePattern,
    /// Source representation.
    pub src_repr: Representation,
    /// Destination representation (equal to `src_repr` unless the engine converts).
    pub dst_repr: Representation,
}

/// Capability matrix of an engine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Capabilities {
    routes: Vec<RouteCap>,
}

impl Capabilities {
    /// Empty matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Identity routes (no conversion) for each listed representation.
    pub fn allow(
        mut self,
        src: impl Into<DevicePattern>,
        dst: impl Into<DevicePattern>,
        representations: &[Representation],
    ) -> Self {
        let (src, dst) = (src.into(), dst.into());
        for &r in representations {
            self = self.allow_conversion(src, dst, r, r);
        }
        self
    }

    /// A route that converts `from` to `to` on the way.
    pub fn allow_conversion(
        mut self,
        src: impl Into<DevicePattern>,
        dst: impl Into<DevicePattern>,
        from: Representation,
        to: Representation,
    ) -> Self {
        let cap = RouteCap {
            src: src.into(),
            dst: dst.into(),
            src_repr: from,
            dst_repr: to,
        };
        if !self.routes.contains(&cap) {
            self.routes.push(cap);
        }
        self
    }

    /// True if some declared route covers this one.
    pub fn supports(
        &self,
        src: Location,
        dst: Location,
        src_repr: Representation,
        dst_repr: Representation,
    ) -> bool {
        self.routes.iter().any(|c| {
            c.src.matches(src)
                && c.dst.matches(dst)
                && c.src_repr == src_repr
                && c.dst_repr == dst_repr
        })
    }

    /// All declared routes.
    pub fn routes(&self) -> &[RouteCap] {
        &self.routes
    }
}

/// An explicitly selected engine route between two device locations.
#[derive(Clone)]
pub struct Route {
    engine: Arc<dyn CopyEngine>,
    src: Location,
    dst: Location,
    src_repr: Representation,
    dst_repr: Representation,
}

impl fmt::Debug for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Route({}: {} -> {}, {} -> {})",
            self.engine.name(),
            self.src,
            self.dst,
            self.src_repr,
            self.dst_repr
        )
    }
}

/// Ids of submitted jobs, unique per process (they key byte-range claims on regions).
static NEXT_JOB: AtomicU64 = AtomicU64::new(1);

impl Route {
    /// Select an identity route (no conversion). Fails with [`Error::Unsupported`] if the
    /// engine does not serve it.
    pub fn select(
        engine: Arc<dyn CopyEngine>,
        src: impl Into<Location>,
        dst: impl Into<Location>,
        repr: Representation,
    ) -> Result<Route> {
        Self::select_converting(engine, src, dst, repr, repr)
    }

    /// Select a route that writes `dst_repr` from `src_repr` data. Fails with
    /// [`Error::Unsupported`] unless the engine declared exactly this conversion.
    pub fn select_converting(
        engine: Arc<dyn CopyEngine>,
        src: impl Into<Location>,
        dst: impl Into<Location>,
        src_repr: Representation,
        dst_repr: Representation,
    ) -> Result<Route> {
        let (src, dst) = (src.into(), dst.into());
        if src.node.is_some() || dst.node.is_some() {
            return Err(Error::Unsupported(format!(
                "route {src} -> {dst} involves another node; network engines arrive in phase P3"
            )));
        }
        if !engine.capabilities().supports(src, dst, src_repr, dst_repr) {
            return Err(Error::Unsupported(format!(
                "engine {} has no route {src} -> {dst} for {src_repr} -> {dst_repr}",
                engine.name()
            )));
        }
        Ok(Route {
            engine,
            src,
            dst,
            src_repr,
            dst_repr,
        })
    }

    /// Source location.
    pub fn src(&self) -> Location {
        self.src
    }
    /// Destination location.
    pub fn dst(&self) -> Location {
        self.dst
    }
    /// Source representation.
    pub fn representation(&self) -> Representation {
        self.src_repr
    }
    /// Destination representation.
    pub fn dst_representation(&self) -> Representation {
        self.dst_repr
    }
    /// True if the route converts representations.
    pub fn converts(&self) -> bool {
        self.src_repr != self.dst_repr
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
        // Claim the byte ranges every span touches, per distinct region. Disjoint claims of
        // other jobs on the same regions are fine; an overlap rejects the whole job unchanged.
        let mut claims: Vec<(RegionRef, Vec<(u64, u64)>)> = Vec::new();
        let mut claim =
            |r: &RegionRef, a: u64, len: u64| match claims.iter_mut().find(|(x, _)| x.same(r)) {
                Some((_, ranges)) => ranges.push((a, a + len)),
                None => claims.push((r.clone(), vec![(a, a + len)])),
            };
        for s in spans {
            claim(&src[s.src as usize], s.src_offset, s.len);
            claim(&dst[s.dst as usize], s.dst_offset, self.dst_len(s.len));
        }
        let id = NEXT_JOB.fetch_add(1, Ordering::Relaxed);
        for (i, (r, ranges)) in claims.iter().enumerate() {
            if let Err(e) = r.begin(id, ranges) {
                for (r, _) in &claims[..i] {
                    r.rollback(id);
                }
                return Err(e);
            }
        }
        let regions = claims.into_iter().map(|(r, _)| r).collect();
        let job = CopyJob {
            src,
            dst,
            spans,
            representation: self.src_repr,
            dst_representation: self.dst_repr,
            direction,
        };
        let bytes = job.bytes();
        // From here on the transfer owns the claims' fate: drop or failure means Uncertain.
        let mut transfer = Transfer {
            id,
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

    fn dst_len(&self, len: u64) -> u64 {
        self.src_repr.convert_len(len, self.dst_repr)
    }

    fn validate(&self, src: &[RegionRef], dst: &[RegionRef], spans: &[Span]) -> Result<()> {
        if spans.is_empty() {
            return invalid("empty copy plan");
        }
        for (side, regions, loc) in [("source", src, self.src), ("destination", dst, self.dst)] {
            if let Some(r) = regions.iter().find(|r| r.location() != loc) {
                return Err(Error::Unsupported(format!(
                    "{side} region is at {} but the route expects {loc}",
                    r.location()
                )));
            }
        }
        let mut writes: Vec<(usize, u64, u64)> = Vec::with_capacity(spans.len());
        for s in spans {
            let (Some(a), Some(b)) = (src.get(s.src as usize), dst.get(s.dst as usize)) else {
                return invalid("span names an unknown region");
            };
            let dst_len = self.dst_len(s.len);
            let in_bounds = |off: u64, n: u64, len: usize| {
                off.checked_add(n).is_some_and(|end| end <= len as u64)
            };
            if s.len == 0
                || s.len % self.src_repr.element_bytes() != 0
                || !in_bounds(s.src_offset, s.len, a.len())
                || !in_bounds(s.dst_offset, dst_len, b.len())
            {
                return invalid("span is empty, splits an element, or lies outside its region");
            }
            writes.push((s.dst as usize, s.dst_offset, s.dst_offset + dst_len));
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

/// Handle to a submitted job. Its byte ranges stay claimed (`InFlight`) until it completes;
/// dropping it before completion marks its regions `Uncertain` (the engine may still be copying).
pub struct Transfer {
    id: u64,
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
    /// Planned source bytes.
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
            r.end(self.id, true);
        }
        Ok(self.bytes)
    }

    fn fail(&mut self, msg: String) -> Error {
        self.completion = None;
        self.outcome = Outcome::Failed(msg.clone());
        for r in &self.regions {
            r.end(self.id, false);
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
    use crate::memory::{Lent, Phase, Region};
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
        let h = Location::HOST;
        assert!(Capabilities::new()
            .allow(MemoryKind::Host, MemoryKind::Host, &[Representation::Fp16])
            .supports(h, h, Representation::Fp16, Representation::Fp16));
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
        let cuda =
            unsafe { Region::external(Location::new(MemoryKind::Cuda, 0), 1, 64, None).unwrap() }
                .lend();
        let e = route
            .submit(&[a.share()], &[cuda.share()], &[ok], Direction::Write)
            .unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)));
    }

    #[test]
    fn disjoint_jobs_share_a_region_overlaps_are_rejected_unchanged() {
        let (a, b) = host_pair(64);
        let caps =
            Capabilities::new().allow(MemoryKind::Host, MemoryKind::Host, &Representation::ALL);
        let route = Route::select(
            Arc::new(Scripted {
                caps,
                result: Mutex::new(None),
                quiesce_ok: true,
            }),
            MemoryKind::Host,
            MemoryKind::Host,
            Representation::Bf16,
        )
        .unwrap();
        let span = |off: u64| Span {
            src: 0,
            src_offset: off,
            dst: 0,
            dst_offset: off,
            len: 16,
        };
        let (sa, sb) = (&[a.share()], &[b.share()]);
        let mut t1 = route.submit(sa, sb, &[span(0)], Direction::Write).unwrap();
        let mut t2 = route.submit(sa, sb, &[span(16)], Direction::Write).unwrap();
        let e = route
            .submit(sa, sb, &[span(32), span(8)], Direction::Write)
            .unwrap_err();
        assert!(matches!(e, Error::Busy(_)));
        // Nothing of the rejected job was claimed: its free range is still available.
        let mut t3 = route.submit(sa, sb, &[span(32)], Direction::Write).unwrap();
        for t in [&mut t1, &mut t2] {
            t.wait().unwrap();
        }
        assert_eq!(b.phase(), Phase::InFlight);
        t3.wait().unwrap();
        assert_eq!(b.phase(), Phase::Done);
    }

    #[test]
    fn device_indices_and_conversions_are_explicit() {
        let caps = Capabilities::new()
            .allow(MemoryKind::Cuda, MemoryKind::Cuda, &[Representation::Bf16])
            .allow_conversion(
                DevicePattern::at(MemoryKind::Rngd, 3),
                DevicePattern::any(MemoryKind::Atom),
                Representation::Bf16,
                Representation::Dlfloat16,
            );
        let engine: Arc<dyn CopyEngine> = Arc::new(Scripted {
            caps,
            result: Mutex::new(None),
            quiesce_ok: true,
        });
        let loc = |s: &str| s.parse::<Location>().unwrap();
        let bf16 = Representation::Bf16;
        let dlf = Representation::Dlfloat16;
        assert!(Route::select(engine.clone(), loc("cuda:0"), loc("cuda:1"), bf16).is_ok());
        let r = Route::select_converting(engine.clone(), loc("rngd:3"), loc("atom:2"), bf16, dlf)
            .unwrap();
        assert!(r.converts() && r.dst() == loc("atom:2"));
        // Not declared: other RNGD device, identity on the converting pair, reverse conversion.
        for (src, dst, a, b) in [
            ("rngd:0", "atom:2", bf16, dlf),
            ("rngd:3", "atom:2", bf16, bf16),
            ("rngd:3", "atom:2", dlf, bf16),
        ] {
            let e = Route::select_converting(engine.clone(), loc(src), loc(dst), a, b).unwrap_err();
            assert!(matches!(e, Error::Unsupported(_)));
        }
        // Cross-node routes need a network engine.
        let e = Route::select(engine.clone(), loc("cuda:0"), loc("cuda:0@7"), bf16).unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)));
        // A region on another device of the same kind does not fit the route.
        let (a, _) = host_pair(8);
        let r01 = Route::select(engine, loc("cuda:0"), loc("cuda:1"), bf16).unwrap();
        let wrong = unsafe { Region::external(loc("cuda:2"), 1, 64, None).unwrap() }.lend();
        let src = unsafe { Region::external(loc("cuda:0"), 1, 64, None).unwrap() }.lend();
        let span = Span {
            src: 0,
            src_offset: 0,
            dst: 0,
            dst_offset: 0,
            len: 8,
        };
        let e = r01
            .submit(&[src.share()], &[wrong.share()], &[span], Direction::Write)
            .unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)));
        drop(a);
        assert_eq!(
            "cuda:*".parse::<DevicePattern>().unwrap(),
            DevicePattern::any(MemoryKind::Cuda)
        );
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
        fn submit(&self, job: &CopyJob<'_>) -> Result<Box<dyn Completion>> {
            // No scripted result: complete honestly with the job's byte count.
            match self.result.lock().unwrap().take() {
                Some(Err(e)) if e.is_rejection() => Err(Error::Transfer(e.to_string())),
                None => Ok(Box::new(ScriptedCompletion(Some(Ok(job.bytes()))))),
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
