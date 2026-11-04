//! L1 memory: regions, their ownership state machine, and registration.
//!
//! A region is a byte range at one [`Location`] (memory kind, device index, optional node). Its
//! owner moves it through
//!
//! ```text
//!   Owned --lend--> Lent --submit--> InFlight --all fenced ok--> Done --reclaim--> Owned
//!                     ^                  |                        |
//!                     |                  +--error/drop----------> Uncertain --quiesce+reclaim--> Owned
//!                     +------------- (Done may be submitted again) ---+
//! ```
//!
//! Ownership of in-flight bytes is tracked per byte range: many copies may be in flight on one
//! region (e.g. a paged KV pool shared by many requests) as long as their ranges are disjoint;
//! an overlapping submission is rejected with `Busy`. One uncertain completion taints the whole
//! region until the engine is quiesced (conservative: a device that failed may have written
//! anywhere it was allowed to).
//!
//! The types enforce the rules:
//!
//! * [`Region`] is the unique owner and the only way to touch host bytes. It is not `Clone`.
//! * [`Region::lend`] consumes it and yields a [`Lent`] handle. Transfers and sessions hold
//!   [`RegionRef`]s obtained from [`Lent::share`]; while any `RegionRef` is alive (e.g. inside a
//!   transfer handle or a poisoned session), [`Lent::reclaim`] fails.
//! * A region whose transfer failed or was abandoned is `Uncertain`; it can only be reclaimed
//!   with a [`Quiesced`] proof, which only a successful engine quiesce produces.
//! * Memory still `Uncertain` when the last reference drops is leaked, never freed, so a device
//!   that might still be writing to it can never corrupt a reallocation.

use std::alloc::{self, Layout};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde::{Deserialize, Serialize};

use crate::error::{invalid, Error, Result};

/// Memory tier of a region. P0 engines only move `Host`/`HostPinned` bytes themselves; the
/// device and storage kinds exist so routes and descriptors can name them (Python-provided
/// engines may already serve them, e.g. a Torch engine for `Cuda`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Pageable host memory.
    Host,
    /// Page-locked host memory (registered with a device driver by its allocator).
    HostPinned,
    /// NVIDIA GPU memory.
    Cuda,
    /// FuriosaAI RNGD device memory.
    Rngd,
    /// Rebellions ATOM device memory.
    Atom,
    /// Local storage tier (file / NVMe). Placeholder: no engine in P0.
    Storage,
    /// Memory owned by a remote agent / network store. Placeholder: no engine in P0.
    Remote,
}

impl MemoryKind {
    /// All kinds, in declaration order.
    pub const ALL: [MemoryKind; 7] = [
        MemoryKind::Host,
        MemoryKind::HostPinned,
        MemoryKind::Cuda,
        MemoryKind::Rngd,
        MemoryKind::Atom,
        MemoryKind::Storage,
        MemoryKind::Remote,
    ];

    /// True for kinds whose addresses are host virtual addresses this process may dereference.
    pub fn is_host(self) -> bool {
        matches!(self, MemoryKind::Host | MemoryKind::HostPinned)
    }

    /// Stable lowercase name, also used on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Host => "host",
            MemoryKind::HostPinned => "host_pinned",
            MemoryKind::Cuda => "cuda",
            MemoryKind::Rngd => "rngd",
            MemoryKind::Atom => "atom",
            MemoryKind::Storage => "storage",
            MemoryKind::Remote => "remote",
        }
    }

    /// The `medium` string used in vLLM-style KV events.
    pub fn medium(self) -> &'static str {
        match self {
            MemoryKind::Host | MemoryKind::HostPinned => "CPU",
            MemoryKind::Cuda => "GPU",
            MemoryKind::Rngd => "RNGD",
            MemoryKind::Atom => "ATOM",
            MemoryKind::Storage => "DISK",
            MemoryKind::Remote => "REMOTE",
        }
    }
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryKind {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        MemoryKind::ALL
            .into_iter()
            .find(|k| k.as_str() == s)
            .ok_or_else(|| Error::Invalid(format!("unknown memory kind {s:?}")))
    }
}

/// Where a region lives: memory kind, device index within the kind, and an optional node id
/// (an opaque host identity for cross-host routes in later phases; `None` = this node).
/// Written `kind[:index][@node]`, e.g. `host`, `cuda:1`, `rngd:3`, `atom:2@17`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location {
    /// Memory kind (tier).
    pub kind: MemoryKind,
    /// Device index within the kind (0 for host memory).
    pub index: u32,
    /// Node id for cross-host routes; `None` is the local node.
    pub node: Option<u64>,
}

impl Location {
    /// Pageable host memory of this node.
    pub const HOST: Location = Location::new(MemoryKind::Host, 0);

    /// Local location.
    pub const fn new(kind: MemoryKind, index: u32) -> Location {
        Location {
            kind,
            index,
            node: None,
        }
    }

    /// The same device on node `node`.
    pub const fn on_node(self, node: u64) -> Location {
        Location {
            node: Some(node),
            ..self
        }
    }
}

impl From<MemoryKind> for Location {
    fn from(kind: MemoryKind) -> Self {
        Location::new(kind, 0)
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.index)?;
        if let Some(n) = self.node {
            write!(f, "@{n}")?;
        }
        Ok(())
    }
}

impl FromStr for Location {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let bad = || Error::Invalid(format!("bad location {s:?} (expected kind[:index][@node])"));
        let (dev, node) = match s.split_once('@') {
            Some((d, n)) => (d, Some(n.parse::<u64>().map_err(|_| bad())?)),
            None => (s, None),
        };
        let (kind, index) = match dev.split_once(':') {
            Some((k, i)) => (k, i.parse::<u32>().map_err(|_| bad())?),
            None => (dev, 0),
        };
        Ok(Location {
            kind: kind.parse()?,
            index,
            node,
        })
    }
}

/// Ownership phase of a region (see the module diagram).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Held by its [`Region`] owner; host bytes are accessible.
    Owned,
    /// Lent out, no copy outstanding.
    Lent,
    /// At least one copy touching it was submitted and is not yet fenced.
    InFlight,
    /// Every copy so far completed and was verified.
    Done,
    /// A copy failed or was abandoned: contents and device activity are unknown.
    Uncertain,
}

impl Phase {
    /// Stable lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Owned => "owned",
            Phase::Lent => "lent",
            Phase::InFlight => "in_flight",
            Phase::Done => "done",
            Phase::Uncertain => "uncertain",
        }
    }
}

/// Proof that an engine was quiesced: no copy it accepted is still running.
/// Only [`crate::engine::Route::quiesce`] creates one.
#[derive(Debug)]
pub struct Quiesced {
    _private: (),
}

impl Quiesced {
    pub(crate) fn new() -> Self {
        Quiesced { _private: () }
    }
}

enum Backing {
    /// Allocated by this crate at `addr` with this layout.
    Host(Layout),
    /// Memory owned elsewhere; the keepalive pins it (e.g. a Python tensor).
    External(Option<Box<dyn Any + Send + Sync>>),
}

/// Ownership state. `inflight` holds `(job, start, end)` byte ranges of unfenced copies.
struct State {
    owned: bool,
    holders: usize,
    inflight: Vec<(u64, u64, u64)>,
    done: bool,
    uncertain: bool,
}

impl State {
    fn phase(&self) -> Phase {
        if self.owned {
            Phase::Owned
        } else if self.uncertain {
            Phase::Uncertain
        } else if !self.inflight.is_empty() {
            Phase::InFlight
        } else if self.done {
            Phase::Done
        } else {
            Phase::Lent
        }
    }
}

struct Shared {
    location: Location,
    addr: usize,
    len: usize,
    backing: Backing,
    state: Mutex<State>,
}

impl Shared {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic while holding this lock leaves only plain data; keep going.
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let uncertain = self.state().uncertain;
        match &mut self.backing {
            Backing::Host(layout) if !uncertain => {
                // SAFETY: allocated in `Region::host` with exactly this layout.
                unsafe { alloc::dealloc(self.addr as *mut u8, *layout) }
            }
            Backing::External(keep) if uncertain => {
                // Never release memory a device may still write to.
                std::mem::forget(keep.take());
            }
            _ => {}
        }
    }
}

const HOST_ALIGN: usize = 4096;

/// The unique owner of a region. Only the owner can read or write host bytes.
pub struct Region {
    shared: Arc<Shared>,
}

impl Region {
    /// Allocate `len` zeroed, page-aligned bytes of pageable host memory.
    pub fn host(len: usize) -> Result<Region> {
        if len == 0 {
            return invalid("region length must be positive");
        }
        let layout = Layout::from_size_align(len, HOST_ALIGN)
            .map_err(|_| Error::Invalid("region too large".into()))?;
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(Error::Invalid(format!("cannot allocate {len} bytes")));
        }
        Ok(Self::from_parts(
            Location::HOST,
            ptr as usize,
            len,
            Backing::Host(layout),
        ))
    }

    /// Wrap memory owned elsewhere. `keepalive` is dropped when the region is released.
    ///
    /// # Safety
    /// For local host locations, `addr..addr+len` must be valid, writable memory that stays
    /// alive as long as `keepalive`, and nothing else may access it while the region is lent.
    /// Otherwise `addr` is opaque to this crate and interpreted only by engines serving the
    /// location.
    pub unsafe fn external(
        location: Location,
        addr: usize,
        len: usize,
        keepalive: Option<Box<dyn Any + Send + Sync>>,
    ) -> Result<Region> {
        if len == 0 {
            return invalid("region length must be positive");
        }
        if location.kind.is_host() && location.node.is_none() && addr == 0 {
            return invalid("host region address must be non-null");
        }
        Ok(Self::from_parts(
            location,
            addr,
            len,
            Backing::External(keepalive),
        ))
    }

    fn from_parts(location: Location, addr: usize, len: usize, backing: Backing) -> Region {
        Region {
            shared: Arc::new(Shared {
                location,
                addr,
                len,
                backing,
                state: Mutex::new(State {
                    owned: true,
                    holders: 0,
                    inflight: Vec::new(),
                    done: false,
                    uncertain: false,
                }),
            }),
        }
    }

    /// Location.
    pub fn location(&self) -> Location {
        self.shared.location
    }

    /// Memory kind (tier).
    pub fn kind(&self) -> MemoryKind {
        self.shared.location.kind
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.shared.len
    }

    /// Always false (regions have positive length); provided for API symmetry.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Address (host virtual address for host kinds, opaque otherwise).
    pub fn addr(&self) -> usize {
        self.shared.addr
    }

    fn local_host(&self) -> bool {
        self.kind().is_host() && self.location().node.is_none()
    }

    /// Host bytes, or `None` unless the region is local host memory.
    pub fn as_slice(&self) -> Option<&[u8]> {
        // SAFETY: the owner has exclusive access (see `external`'s contract).
        self.local_host()
            .then(|| unsafe { std::slice::from_raw_parts(self.addr() as *const u8, self.len()) })
    }

    /// Mutable host bytes, or `None` unless the region is local host memory.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        // SAFETY: `&mut self` on the unique owner.
        self.local_host()
            .then(|| unsafe { std::slice::from_raw_parts_mut(self.addr() as *mut u8, self.len()) })
    }

    /// Lend the region out (Owned -> Lent). Get it back with [`Lent::reclaim`].
    pub fn lend(self) -> Lent {
        self.shared.state().owned = false;
        Lent {
            shared: self.shared,
        }
    }
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Region({}, {} bytes)", self.location(), self.len())
    }
}

/// A failed reclaim: the handle comes back together with the reason.
pub struct Reclaim {
    /// The handle, still lent.
    pub lent: Lent,
    /// Why the region cannot be reclaimed yet.
    pub error: Error,
}

impl fmt::Debug for Reclaim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reclaim({})", self.error)
    }
}

/// The lender's handle to a lent region.
pub struct Lent {
    shared: Arc<Shared>,
}

impl Lent {
    /// Location.
    pub fn location(&self) -> Location {
        self.shared.location
    }

    /// Memory kind (tier).
    pub fn kind(&self) -> MemoryKind {
        self.shared.location.kind
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.shared.len
    }

    /// Always false; see [`Region::is_empty`].
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Current phase.
    pub fn phase(&self) -> Phase {
        self.shared.state().phase()
    }

    /// A shared reference for transfers, sessions and registries.
    pub fn share(&self) -> RegionRef {
        RegionRef::new(self.shared.clone())
    }

    /// Get the owner back. Fails while any [`RegionRef`] is alive, a copy is in flight, or the
    /// region is uncertain.
    pub fn reclaim(self) -> std::result::Result<Region, Reclaim> {
        self.reclaim_inner(false)
    }

    /// Like [`Lent::reclaim`], but also accepts an `Uncertain` region after the engine that
    /// touched it was quiesced. The contents are unspecified.
    pub fn reclaim_quiesced(self, _proof: &Quiesced) -> std::result::Result<Region, Reclaim> {
        self.reclaim_inner(true)
    }

    fn reclaim_inner(self, quiesced: bool) -> std::result::Result<Region, Reclaim> {
        let error = {
            let mut st = self.shared.state();
            let error = match (st.holders, st.phase()) {
                (0, Phase::Lent | Phase::Done) => None,
                (0, Phase::Uncertain) if quiesced => None,
                (0, Phase::Uncertain) => Some(Error::State(
                    "region is uncertain; quiesce the engine before reclaiming".into(),
                )),
                (0, phase) => Some(Error::State(format!("region is {}", phase.as_str()))),
                (n, _) => Some(Error::Busy(format!(
                    "region is still referenced by {n} transfer/session/registry handle(s)"
                ))),
            };
            if error.is_none() {
                *st = State {
                    owned: true,
                    holders: 0,
                    inflight: Vec::new(),
                    done: false,
                    uncertain: false,
                };
            }
            error
        };
        match error {
            None => Ok(Region {
                shared: self.shared,
            }),
            Some(error) => Err(Reclaim { lent: self, error }),
        }
    }
}

impl fmt::Debug for Lent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lent({}, {} bytes, {})",
            self.location(),
            self.len(),
            self.phase().as_str()
        )
    }
}

/// A counted reference to a lent region, held by transfers, sessions and registries.
pub struct RegionRef {
    shared: Arc<Shared>,
}

impl RegionRef {
    fn new(shared: Arc<Shared>) -> Self {
        shared.state().holders += 1;
        RegionRef { shared }
    }

    /// Location.
    pub fn location(&self) -> Location {
        self.shared.location
    }

    /// Memory kind (tier).
    pub fn kind(&self) -> MemoryKind {
        self.shared.location.kind
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.shared.len
    }

    /// Always false; see [`Region::is_empty`].
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Address (host virtual address for host kinds, opaque otherwise).
    pub fn addr(&self) -> usize {
        self.shared.addr
    }

    /// Current phase.
    pub fn phase(&self) -> Phase {
        self.shared.state().phase()
    }

    /// The keepalive object of an external region (engines downcast it, e.g. to a tensor).
    pub fn keepalive(&self) -> Option<&(dyn Any + Send + Sync)> {
        match &self.shared.backing {
            Backing::External(Some(k)) => Some(k.as_ref()),
            _ => None,
        }
    }

    /// True when both references name the same region.
    pub fn same(&self, other: &RegionRef) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// Claim byte ranges `[start, end)` for copy `job`. Fails with `Busy` if any range overlaps
    /// a range of another unfenced copy, and with `State` if the region is owned or uncertain;
    /// either way nothing changes.
    pub(crate) fn begin(&self, job: u64, ranges: &[(u64, u64)]) -> Result<()> {
        let mut st = self.shared.state();
        if st.owned {
            return Err(Error::State("region is not lent".into()));
        }
        if st.uncertain {
            return Err(Error::State(
                "region is uncertain; quiesce before reuse".into(),
            ));
        }
        for &(a, b) in ranges {
            if let Some(&(_, c, d)) = st.inflight.iter().find(|&&(_, c, d)| a < d && c < b) {
                return Err(Error::Busy(format!(
                    "bytes {a}..{b} overlap bytes {c}..{d} of an unfenced copy on this region"
                )));
            }
        }
        st.inflight.extend(ranges.iter().map(|&(a, b)| (job, a, b)));
        Ok(())
    }

    /// Undo a `begin` when submission is rejected before reaching the engine.
    pub(crate) fn rollback(&self, job: u64) {
        self.shared.state().inflight.retain(|r| r.0 != job);
    }

    /// Copy `job` finished: release its ranges; a failure taints the whole region.
    pub(crate) fn end(&self, job: u64, ok: bool) {
        let mut st = self.shared.state();
        st.inflight.retain(|r| r.0 != job);
        if !ok {
            st.uncertain = true;
        } else if st.inflight.is_empty() {
            st.done = true;
        }
    }

    /// Uncertain -> Lent after a quiesce.
    pub(crate) fn settle(&self, _proof: &Quiesced) {
        let mut st = self.shared.state();
        if st.uncertain {
            st.uncertain = false;
            st.done = false;
        }
    }
}

impl Clone for RegionRef {
    fn clone(&self) -> Self {
        RegionRef::new(self.shared.clone())
    }
}

impl Drop for RegionRef {
    fn drop(&mut self) {
        self.shared.state().holders -= 1;
    }
}

impl fmt::Debug for RegionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RegionRef({}, {} bytes, {})",
            self.location(),
            self.len(),
            self.phase().as_str()
        )
    }
}

/// Serializable name of a registered region: what a peer needs to address it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionDesc {
    /// Registry (agent memory domain) that owns the region.
    pub registry: u64,
    /// Region id within that registry.
    pub region: u64,
    /// Where the region lives.
    pub location: Location,
    /// Length in bytes.
    pub len: u64,
}

impl RegionDesc {
    /// Versioned wire encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::wire::encode("region", self)
    }

    /// Decode [`RegionDesc::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        crate::wire::decode("region", bytes)
    }
}

struct RegistryInner {
    id: u64,
    next: AtomicU64,
    regions: Mutex<HashMap<u64, RegionRef>>,
}

impl Drop for RegistryInner {
    fn drop(&mut self) {
        directory()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.id);
    }
}

type Directory = Mutex<HashMap<u64, Weak<RegistryInner>>>;

/// Process-wide map of live registries, so descriptors resolve across agents in one process.
fn directory() -> &'static Directory {
    static DIR: OnceLock<Directory> = OnceLock::new();
    DIR.get_or_init(Default::default)
}

fn fresh_id() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(std::process::id());
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    if let Ok(t) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        h.write_u128(t.as_nanos());
    }
    h.finish()
}

/// Registered memory of one agent. Registering a region holds a [`RegionRef`] (so the owner
/// cannot reclaim it) until it is deregistered.
#[derive(Clone)]
pub struct Registry {
    inner: Arc<RegistryInner>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// A new registry with a random 64-bit id.
    pub fn new() -> Registry {
        let mut dir = directory().lock().unwrap_or_else(|p| p.into_inner());
        let mut id = fresh_id();
        while dir.contains_key(&id) {
            id = fresh_id();
        }
        let inner = Arc::new(RegistryInner {
            id,
            next: AtomicU64::new(1),
            regions: Mutex::new(HashMap::new()),
        });
        dir.insert(id, Arc::downgrade(&inner));
        Registry { inner }
    }

    /// Registry id carried in descriptors.
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    fn regions(&self) -> std::sync::MutexGuard<'_, HashMap<u64, RegionRef>> {
        self.inner.regions.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Register a region and return its descriptor.
    pub fn register(&self, region: &RegionRef) -> RegionDesc {
        let id = self.inner.next.fetch_add(1, Ordering::Relaxed);
        self.regions().insert(id, region.clone());
        RegionDesc {
            registry: self.id(),
            region: id,
            location: region.location(),
            len: region.len() as u64,
        }
    }

    /// Drop the registration (and its reference).
    pub fn deregister(&self, desc: &RegionDesc) -> Result<()> {
        if desc.registry != self.id() || self.regions().remove(&desc.region).is_none() {
            return invalid("region is not registered here");
        }
        Ok(())
    }

    /// Number of registered regions.
    pub fn len(&self) -> usize {
        self.regions().len()
    }

    /// True when nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve a descriptor to the region it names.
    ///
    /// P0 resolves descriptors of registries that live in this process. A descriptor from
    /// another process is `Unsupported` until cross-process transport (P3) exists.
    pub fn resolve(desc: &RegionDesc) -> Result<RegionRef> {
        let inner = directory()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&desc.registry)
            .and_then(Weak::upgrade)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "registry {:#x} is not in this process; cross-process transport is phase P3",
                    desc.registry
                ))
            })?;
        let reg = Registry { inner };
        let regions = reg.regions();
        let r = regions
            .get(&desc.region)
            .ok_or_else(|| Error::Invalid("descriptor names an unregistered region".into()))?;
        if r.location() != desc.location || r.len() as u64 != desc.len {
            return invalid("descriptor location/length differ from the registered region");
        }
        Ok(r.clone())
    }
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Registry({:#x}, {} regions)", self.id(), self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_region_is_zeroed_and_writable() {
        let mut r = Region::host(100).unwrap();
        assert!(r.as_slice().unwrap().iter().all(|&b| b == 0));
        r.as_mut_slice().unwrap()[3] = 7;
        assert_eq!(r.as_slice().unwrap()[3], 7);
        assert_eq!(r.addr() % HOST_ALIGN, 0);
        assert!(Region::host(0).is_err());
    }

    #[test]
    fn reclaim_requires_no_holders_and_settled_phase() {
        let lent = Region::host(16).unwrap().lend();
        assert_eq!(lent.phase(), Phase::Lent);
        let r = lent.share();
        let lent = match lent.reclaim() {
            Err(Reclaim { lent, error }) => {
                assert!(matches!(error, Error::Busy(_)));
                lent
            }
            Ok(_) => panic!("reclaimed while referenced"),
        };
        r.begin(1, &[(0, 16)]).unwrap();
        assert_eq!(lent.phase(), Phase::InFlight);
        r.end(1, true);
        assert_eq!(lent.phase(), Phase::Done);
        drop(r);
        let region = lent.reclaim().unwrap();
        assert!(region.as_slice().is_some());
    }

    #[test]
    fn disjoint_ranges_run_concurrently_overlaps_are_busy() {
        let lent = Region::host(64).unwrap().lend();
        let r = lent.share();
        r.begin(1, &[(0, 16), (32, 40)]).unwrap();
        r.begin(2, &[(16, 32)]).unwrap();
        // Overlap with job 1: rejected, and job 3 claims nothing (not even its free range).
        assert!(matches!(
            r.begin(3, &[(48, 64), (39, 41)]),
            Err(Error::Busy(_))
        ));
        r.begin(4, &[(48, 64)]).unwrap();
        r.end(1, true);
        r.end(2, true);
        assert_eq!(r.phase(), Phase::InFlight);
        r.end(4, true);
        assert_eq!(r.phase(), Phase::Done);
        r.begin(5, &[(0, 64)]).unwrap();
        r.rollback(5);
        assert_eq!(r.phase(), Phase::Done);
    }

    #[test]
    fn uncertain_taints_whole_region_until_quiesce() {
        let lent = Region::host(64).unwrap().lend();
        let r = lent.share();
        r.begin(1, &[(0, 8)]).unwrap();
        r.begin(2, &[(8, 16)]).unwrap();
        r.end(1, false);
        assert_eq!(r.phase(), Phase::Uncertain);
        assert!(
            r.begin(3, &[(32, 40)]).is_err(),
            "disjoint bytes are tainted too"
        );
        r.end(2, true);
        assert_eq!(r.phase(), Phase::Uncertain);
        drop(r);
        let lent = lent.reclaim().unwrap_err().lent;
        let proof = Quiesced::new();
        let region = lent.reclaim_quiesced(&proof).unwrap();
        assert_eq!(region.len(), 64);
    }

    #[test]
    fn owned_region_cannot_start_a_copy() {
        let lent = Region::host(8).unwrap().lend();
        let r = lent.share();
        r.settle(&Quiesced::new()); // no-op unless uncertain
        assert_eq!(r.phase(), Phase::Lent);
    }

    #[test]
    fn locations_parse_and_print() {
        let l: Location = "rngd:3".parse().unwrap();
        assert_eq!(l, Location::new(MemoryKind::Rngd, 3));
        assert_eq!("host".parse::<Location>().unwrap(), Location::HOST);
        let remote: Location = "atom:2@17".parse().unwrap();
        assert_eq!(remote, Location::new(MemoryKind::Atom, 2).on_node(17));
        assert_eq!(remote.to_string(), "atom:2@17");
        for bad in ["gpu:0", "cuda:x", "cuda:0@", "cuda:-1"] {
            assert!(bad.parse::<Location>().is_err(), "{bad}");
        }
    }

    #[test]
    fn registry_resolves_in_process_and_rejects_foreign() {
        let reg = Registry::new();
        let lent = Region::host(32).unwrap().lend();
        let desc = reg.register(&lent.share());
        let decoded = RegionDesc::from_bytes(&desc.to_bytes()).unwrap();
        assert_eq!(decoded, desc);
        let r = Registry::resolve(&decoded).unwrap();
        assert_eq!(r.len(), 32);
        drop(r);
        // Registration holds a reference.
        let lent = lent.reclaim().unwrap_err().lent;
        reg.deregister(&desc).unwrap();
        assert!(Registry::resolve(&desc).is_err());
        lent.reclaim().unwrap();

        let foreign = RegionDesc {
            registry: reg.id() ^ 1,
            ..desc
        };
        assert!(matches!(
            Registry::resolve(&foreign),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn memory_kind_names_round_trip() {
        for k in MemoryKind::ALL {
            assert_eq!(k.as_str().parse::<MemoryKind>().unwrap(), k);
        }
        assert!("gpu".parse::<MemoryKind>().is_err());
    }
}
