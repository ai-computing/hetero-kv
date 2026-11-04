//! L4 session: the per-destination lifecycle
//!
//! ```text
//!   Idle --reserve--> Reserved --transfer*--> Transferring --fence--> Fenced --admit--> Admitted
//!    ^                   |                         |                    |                 |
//!    |                   +--cancel--> Idle         +--cancel: flag      +--cancel--> Idle  +--finish(ok)--> Idle
//!    |                                             any engine error / failed notify / finish(failed)
//!    +--recover (engine quiesced)-- Poisoned <---------------------------------------------+
//! ```
//!
//! * **Single flight.** One request per session; `reserve` on a busy session fails with `Busy`.
//! * **Completion-fenced admission.** `admit` (which runs the caller's visibility hook, e.g.
//!   marking a decoder's cache ready) is only accepted once `fence` verified every transfer.
//! * **Poison on uncertainty.** An engine failure, an inconsistent completion, a failed admit
//!   hook or `finish(passed = false)` poisons the session. Its buffers are retained (owners cannot
//!   reclaim them) until `recover` quiesces the engine.
//! * **Rejections are free.** Invalid selections, unsupported routes, wrong steps and stale
//!   request ids change nothing.
//! * **Cancel** is cooperative and never aborts device work: before any transfer it releases the
//!   reservation; with transfers in flight it is recorded, `fence` still runs, and `admit` then
//!   discards the request instead of admitting it.
//! * **Async.** `transfer` returns as soon as the engine accepted the job; `poll`, `wait`,
//!   `wait_layer` and `ready_layers` observe completion per transfer or per layer (vLLM
//!   `wait_for_layer_load`, layer streaming). Admission still needs a full `fence`.

mod agent;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

pub use agent::{Agent, AgentMeta};

use crate::engine::{Direction, Route, Transfer};
use crate::error::{invalid, Error, Result};
use crate::kv::{plan, KvBufferRef, Selection};

/// Lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// No request.
    Idle,
    /// Destination reserved, nothing submitted.
    Reserved,
    /// Transfers submitted, not all fenced.
    Transferring,
    /// Every transfer completed and was verified.
    Fenced,
    /// The destination was made visible; the consumer is using it.
    Admitted,
    /// An uncertain failure occurred; `recover` is required.
    Poisoned,
}

impl SessionState {
    /// Lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            SessionState::Idle => "idle",
            SessionState::Reserved => "reserved",
            SessionState::Transferring => "transferring",
            SessionState::Fenced => "fenced",
            SessionState::Admitted => "admitted",
            SessionState::Poisoned => "poisoned",
        }
    }
}

/// Summary of the fenced transfers of one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    /// Request id.
    pub request_id: String,
    /// Engine name.
    pub engine: String,
    /// Number of transfers.
    pub transfers: u32,
    /// Number of spans.
    pub spans: u64,
    /// Bytes copied and verified.
    pub bytes: u64,
}

/// Handle to one submitted transfer within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(pub u64);

struct Pending {
    id: u64,
    layers: Vec<u32>,
    transfer: Transfer,
}

struct Inner {
    state: SessionState,
    request_id: Option<String>,
    dst: Vec<KvBufferRef>,
    src: Vec<KvBufferRef>,
    pending: Vec<Pending>,
    retained: Vec<KvBufferRef>,
    poison: Option<String>,
    next_id: u64,
}

impl Inner {
    fn release(&mut self) {
        self.state = SessionState::Idle;
        self.request_id = None;
        self.dst.clear();
        self.src.clear();
        self.pending.clear();
    }

    /// Poison and retain every buffer. Dropping unfinished transfers marks their regions
    /// uncertain, which is exactly what we know about them.
    fn poison(&mut self, error: Error) -> Error {
        self.state = SessionState::Poisoned;
        self.poison = Some(error.to_string());
        self.retained.append(&mut self.dst);
        self.retained.append(&mut self.src);
        self.pending.clear();
        error
    }

    fn receipt(&self, engine: &str) -> Receipt {
        Receipt {
            request_id: self.request_id.clone().unwrap_or_default(),
            engine: engine.to_owned(),
            transfers: self.pending.len() as u32,
            spans: self.pending.iter().map(|p| p.transfer.spans() as u64).sum(),
            bytes: self.pending.iter().map(|p| p.transfer.bytes()).sum(),
        }
    }
}

/// One destination lease's transfer lifecycle over an explicitly selected route.
pub struct Session {
    route: Route,
    inner: Mutex<Inner>,
    cancelled: AtomicBool,
}

impl Session {
    /// Idle session on `route`.
    pub fn new(route: Route) -> Session {
        Session {
            route,
            inner: Mutex::new(Inner {
                state: SessionState::Idle,
                request_id: None,
                dst: Vec::new(),
                src: Vec::new(),
                pending: Vec::new(),
                retained: Vec::new(),
                poison: None,
                next_id: 1,
            }),
            cancelled: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Route the session copies over.
    pub fn route(&self) -> &Route {
        &self.route
    }

    /// Current state.
    pub fn state(&self) -> SessionState {
        self.lock().state
    }

    /// Request currently holding the session.
    pub fn request_id(&self) -> Option<String> {
        self.lock().request_id.clone()
    }

    /// Why the session was poisoned (cleared by `recover`).
    pub fn poison_reason(&self) -> Option<String> {
        self.lock().poison.clone()
    }

    /// Buffers retained by a poisoned session.
    pub fn retained(&self) -> usize {
        self.lock().retained.len()
    }

    /// Lock and check that `request_id` holds the session in one of `states`.
    fn holding(&self, request_id: &str, states: &[SessionState]) -> Result<MutexGuard<'_, Inner>> {
        let inner = self.lock();
        if inner.state == SessionState::Poisoned {
            return Err(Error::Poisoned(format!(
                "session poisoned ({}); recover required",
                inner.poison.as_deref().unwrap_or("unknown")
            )));
        }
        if inner.request_id.as_deref() != Some(request_id) {
            return Err(Error::State(format!(
                "stale or foreign request {request_id:?}"
            )));
        }
        if !states.contains(&inner.state) {
            return Err(Error::State(format!(
                "request {request_id:?} is {}",
                inner.state.as_str()
            )));
        }
        Ok(inner)
    }

    /// Reserve destination buffers (one per KV group; most models have one) for `request_id`.
    pub fn reserve(&self, request_id: &str, dst: Vec<KvBufferRef>) -> Result<()> {
        if request_id.is_empty() {
            return invalid("request id required");
        }
        if dst.is_empty() {
            return invalid("at least one destination buffer required");
        }
        for d in &dst {
            if d.kind() != self.route.dst()
                || d.spec().representation() != self.route.representation()
            {
                return Err(Error::Unsupported(format!(
                    "destination {} / {} does not match route {:?}",
                    d.kind(),
                    d.spec().representation(),
                    self.route
                )));
            }
        }
        let mut inner = self.lock();
        match inner.state {
            SessionState::Idle => {}
            SessionState::Poisoned => {
                return Err(Error::Poisoned("session poisoned; recover required".into()))
            }
            s => return Err(Error::Busy(format!("session is {}", s.as_str()))),
        }
        self.cancelled.store(false, Ordering::SeqCst);
        inner.state = SessionState::Reserved;
        inner.request_id = Some(request_id.to_owned());
        inner.dst = dst;
        Ok(())
    }

    /// Plan `selections` from `src` into destination group `group` and submit them as one
    /// engine job. Returns once the engine accepted the job.
    pub fn transfer(
        &self,
        request_id: &str,
        group: usize,
        src: &KvBufferRef,
        selections: &[Selection],
        direction: Direction,
    ) -> Result<TransferId> {
        use SessionState::*;
        let mut inner = self.holding(request_id, &[Reserved, Transferring, Fenced])?;
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(Error::State(format!(
                "request {request_id:?} was cancelled"
            )));
        }
        let dst = inner
            .dst
            .get(group)
            .ok_or_else(|| Error::Invalid(format!("no destination group {group}")))?
            .clone();
        if src.spec().representation() != self.route.representation() {
            return Err(Error::Unsupported(format!(
                "source representation {} does not match route {:?}",
                src.spec().representation(),
                self.route
            )));
        }
        if selections.is_empty() {
            return invalid("at least one selection required");
        }
        let mut spans = Vec::new();
        let mut layers = Vec::new();
        for sel in selections {
            let plan = plan(src.spec(), dst.spec(), sel)?;
            spans.extend_from_slice(plan.spans());
            layers.extend(plan.layers());
        }
        layers.sort_unstable();
        layers.dedup();
        let transfer = match self
            .route
            .submit(src.regions(), dst.regions(), &spans, direction)
        {
            Ok(t) => t,
            Err(e) if e.is_rejection() => return Err(e),
            Err(e) => return Err(inner.poison(e)),
        };
        let id = inner.next_id;
        inner.next_id += 1;
        inner.pending.push(Pending {
            id,
            layers,
            transfer,
        });
        inner.src.push(src.clone());
        inner.state = Transferring;
        Ok(TransferId(id))
    }

    /// Non-blocking completion check of one transfer.
    pub fn poll(&self, request_id: &str, id: TransferId) -> Result<bool> {
        self.with_transfer(request_id, id, |t| t.poll())
    }

    /// Wait for one transfer.
    pub fn wait(&self, request_id: &str, id: TransferId) -> Result<()> {
        self.with_transfer(request_id, id, |t| t.wait().map(|_| ()))
    }

    fn with_transfer<T>(
        &self,
        request_id: &str,
        id: TransferId,
        f: impl FnOnce(&mut Transfer) -> Result<T>,
    ) -> Result<T> {
        use SessionState::*;
        let mut inner = self.holding(request_id, &[Transferring, Fenced, Admitted])?;
        let Some(p) = inner.pending.iter_mut().find(|p| p.id == id.0) else {
            return invalid(format!("unknown transfer {}", id.0));
        };
        f(&mut p.transfer).map_err(|e| inner.poison(e))
    }

    /// Wait for every transfer that touches `layer`.
    pub fn wait_layer(&self, request_id: &str, layer: u32) -> Result<()> {
        use SessionState::*;
        let mut inner = self.holding(request_id, &[Reserved, Transferring, Fenced, Admitted])?;
        let mut result = Ok(());
        for p in inner
            .pending
            .iter_mut()
            .filter(|p| p.layers.contains(&layer))
        {
            if let Err(e) = p.transfer.wait() {
                result = Err(e);
                break;
            }
        }
        result.map_err(|e| inner.poison(e))
    }

    /// Layers whose submitted transfers have all completed (polls without blocking).
    pub fn ready_layers(&self, request_id: &str) -> Result<Vec<u32>> {
        use SessionState::*;
        let mut inner = self.holding(request_id, &[Reserved, Transferring, Fenced, Admitted])?;
        let mut touched = std::collections::BTreeMap::<u32, bool>::new();
        let mut failure = None;
        for p in inner.pending.iter_mut() {
            let done = match p.transfer.poll() {
                Ok(d) => d,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            };
            for &l in &p.layers {
                *touched.entry(l).or_insert(true) &= done;
            }
        }
        if let Some(e) = failure {
            return Err(inner.poison(e));
        }
        Ok(touched
            .into_iter()
            .filter(|&(_, d)| d)
            .map(|(l, _)| l)
            .collect())
    }

    /// Wait for and verify every transfer of the request. A request with no transfers (empty
    /// suffix) fences trivially.
    pub fn fence(&self, request_id: &str) -> Result<Receipt> {
        use SessionState::*;
        let mut inner = self.holding(request_id, &[Reserved, Transferring, Fenced])?;
        let mut failure = None;
        for p in inner.pending.iter_mut() {
            if let Err(e) = p.transfer.wait() {
                failure = Some(e);
                break;
            }
        }
        if let Some(e) = failure {
            return Err(inner.poison(e));
        }
        inner.state = Fenced;
        Ok(inner.receipt(self.route.engine_name()))
    }

    /// Make the fenced destination visible by running `notify`, then hand it to the consumer.
    /// If the request was cancelled, the reservation is discarded instead and `Cancelled` is
    /// returned (the copies were fenced, so this is safe).
    pub fn admit(&self, request_id: &str, notify: impl FnOnce() -> Result<()>) -> Result<Receipt> {
        let mut inner = self.holding(request_id, &[SessionState::Fenced])?;
        if self.cancelled.load(Ordering::SeqCst) {
            inner.release();
            return Err(Error::Cancelled(format!(
                "request {request_id:?} cancelled"
            )));
        }
        let receipt = inner.receipt(self.route.engine_name());
        if let Err(e) = notify() {
            return Err(inner.poison(Error::Transfer(format!("admission hook failed: {e}"))));
        }
        inner.state = SessionState::Admitted;
        Ok(receipt)
    }

    /// End an admitted request. `passed = false` (the consumer failed while using the cache)
    /// poisons the session.
    pub fn finish(&self, request_id: &str, passed: bool) -> Result<()> {
        let mut inner = self.holding(request_id, &[SessionState::Admitted])?;
        if !passed {
            return Err(inner.poison(Error::Poisoned(format!(
                "consumer of request {request_id:?} failed; cache cannot be reused"
            ))));
        }
        inner.release();
        Ok(())
    }

    /// Cancel `request_id` (see the module docs). Never aborts device work.
    pub fn cancel(&self, request_id: &str) -> Result<()> {
        use SessionState::*;
        let mut inner = match self.inner.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                // Another thread is waiting on the engine; admission will see the flag.
                self.cancelled.store(true, Ordering::SeqCst);
                return Ok(());
            }
        };
        if inner.request_id.as_deref() != Some(request_id) {
            return Err(Error::State(format!(
                "stale or foreign request {request_id:?}"
            )));
        }
        match inner.state {
            Reserved | Fenced => {
                inner.release();
                Ok(())
            }
            Transferring => {
                self.cancelled.store(true, Ordering::SeqCst);
                Ok(())
            }
            Admitted => Err(Error::State(
                "request is admitted; finish it instead".into(),
            )),
            Poisoned => Err(Error::Poisoned("session poisoned; recover required".into())),
            Idle => Ok(()),
        }
    }

    /// Quiesce the engine, settle every retained region and return to `Idle`. On failure the
    /// session stays poisoned and keeps its buffers.
    pub fn recover(&self) -> Result<()> {
        let mut inner = self.lock();
        match inner.state {
            SessionState::Idle => return Ok(()),
            SessionState::Poisoned => {}
            s => {
                return Err(Error::State(format!(
                    "session is {}, not poisoned",
                    s.as_str()
                )))
            }
        }
        let proof = self.route.quiesce()?;
        for buf in &inner.retained {
            for r in buf.regions() {
                r.settle(&proof);
            }
        }
        inner.retained.clear();
        inner.poison = None;
        inner.release();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
