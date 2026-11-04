use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;
use crate::engine::{Capabilities, Completion, CopyEngine, CopyJob, HostEngine, Representation};
use crate::kv::{Geometry, KvBuffer, KvDescriptor, KvKind, KvSpec, Layout};
use crate::memory::{MemoryKind, Phase, Region};

fn spec(cap: u32) -> KvSpec {
    let g = Geometry {
        representation: Representation::Bf16,
        kv: KvKind::KeyValue,
        layers: 2,
        blocks: 1,
        block_size: cap,
        heads: 2,
        head_dim: 4,
    };
    KvSpec::new("m", "r", Layout::bhsd(g).unwrap()).unwrap()
}

/// A buffer whose bytes are `seed + i` per region.
fn buffer(cap: u32, seed: u8) -> KvBuffer {
    let s = spec(cap);
    let n = s.layout.region_bytes() as usize;
    let regions = (0..s.layout.region_count())
        .map(|r| {
            let mut region = Region::host(n).unwrap();
            for (i, b) in region.as_mut_slice().unwrap().iter_mut().enumerate() {
                *b = seed.wrapping_add((r * 31 + i) as u8);
            }
            region
        })
        .collect();
    KvBuffer::new(s, regions).unwrap()
}

fn host_session() -> Session {
    Session::new(
        Route::select(
            Arc::new(HostEngine::new()),
            MemoryKind::Host,
            MemoryKind::Host,
            Representation::Bf16,
        )
        .unwrap(),
    )
}

/// Engine with controllable completion and failure.
#[derive(Default)]
struct Gate {
    open: AtomicBool,
    fail_wait: AtomicBool,
    fail_quiesce: AtomicBool,
    submits: AtomicUsize,
    caps: Capabilities,
}

impl Gate {
    fn new() -> Arc<Gate> {
        Arc::new(Gate {
            caps: Capabilities::new().allow(
                MemoryKind::Host,
                MemoryKind::Host,
                &Representation::ALL,
            ),
            ..Default::default()
        })
    }
}

struct GateCompletion(Arc<Gate>, u64);
impl Completion for GateCompletion {
    fn poll(&mut self) -> Result<Option<u64>> {
        Ok(self.0.open.load(Ordering::SeqCst).then_some(self.1))
    }
    fn wait(&mut self) -> Result<u64> {
        if self.0.fail_wait.load(Ordering::SeqCst) {
            return Err(Error::Transfer("completion queue reported an error".into()));
        }
        Ok(self.1)
    }
}

struct GateEngine(Arc<Gate>);
impl CopyEngine for GateEngine {
    fn name(&self) -> &str {
        "gate"
    }
    fn capabilities(&self) -> &Capabilities {
        &self.0.caps
    }
    fn submit(&self, job: &CopyJob<'_>) -> Result<Box<dyn Completion>> {
        self.0.submits.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(GateCompletion(self.0.clone(), job.bytes())))
    }
    fn quiesce(&self) -> Result<()> {
        if self.0.fail_quiesce.load(Ordering::SeqCst) {
            Err(Error::Transfer("cannot establish idle device".into()))
        } else {
            Ok(())
        }
    }
}

fn gate_session(gate: &Arc<Gate>) -> Session {
    let engine = Arc::new(GateEngine(gate.clone()));
    Session::new(
        Route::select(
            engine,
            MemoryKind::Host,
            MemoryKind::Host,
            Representation::Bf16,
        )
        .unwrap(),
    )
}

#[test]
fn lifecycle_round_trip_and_reuse() {
    let s = host_session();
    let src = buffer(8, 1);
    let dst = buffer(16, 0);
    for (n, rid) in [(3, "a"), (8, "b"), (1, "c")] {
        s.reserve(rid, vec![dst.share()]).unwrap();
        assert_eq!(s.state(), SessionState::Reserved);
        s.transfer(
            rid,
            0,
            &src.share(),
            &[Selection::prefix(0..2, n)],
            Direction::Write,
        )
        .unwrap();
        let receipt = s.fence(rid).unwrap();
        assert_eq!(receipt.bytes, 2 * 2 * 2 * n * 8);
        assert_eq!(receipt.engine, "host");
        let mut seen = false;
        s.admit(rid, || {
            seen = true;
            Ok(())
        })
        .unwrap();
        assert!(seen);
        s.finish(rid, true).unwrap();
        assert_eq!(s.state(), SessionState::Idle);
    }
    // Verify the last full copy (8 tokens of each head of each region): bhsd rows.
    let src = src.reclaim().unwrap();
    let dst = dst.reclaim().unwrap();
    for (a, b) in src.iter().zip(&dst) {
        let (a, b) = (a.as_slice().unwrap(), b.as_slice().unwrap());
        for head in 0..2 {
            let (sa, sb) = (head * 8 * 8, head * 16 * 8);
            assert_eq!(&a[sa..sa + 64], &b[sb..sb + 64]);
        }
    }
}

#[test]
fn admission_requires_fence() {
    let gate = Gate::new();
    let s = gate_session(&gate);
    let (src, dst) = (buffer(4, 1), buffer(4, 0));
    s.reserve("r", vec![dst.share()]).unwrap();
    s.transfer(
        "r",
        0,
        &src.share(),
        &[Selection::prefix(0..2, 4)],
        Direction::Write,
    )
    .unwrap();
    let e = s.admit("r", || Ok(())).unwrap_err();
    assert!(matches!(e, Error::State(_)));
    assert_eq!(
        s.state(),
        SessionState::Transferring,
        "rejection does not poison"
    );
    gate.open.store(true, Ordering::SeqCst);
    s.fence("r").unwrap();
    s.admit("r", || Ok(())).unwrap();
    s.finish("r", true).unwrap();
}

#[test]
fn engine_failure_poisons_and_retains_until_recovered() {
    let gate = Gate::new();
    let s = gate_session(&gate);
    let (src, dst) = (buffer(4, 1), buffer(4, 0));
    s.reserve("r", vec![dst.share()]).unwrap();
    s.transfer(
        "r",
        0,
        &src.share(),
        &[Selection::prefix(0..2, 4)],
        Direction::Write,
    )
    .unwrap();
    gate.fail_wait.store(true, Ordering::SeqCst);
    assert!(matches!(s.fence("r"), Err(Error::Transfer(_))));
    assert_eq!(s.state(), SessionState::Poisoned);
    assert_eq!(s.retained(), 2);
    assert!(dst.regions().iter().all(|r| r.phase() == Phase::Uncertain));
    assert!(matches!(
        s.reserve("next", vec![dst.share()]),
        Err(Error::Poisoned(_))
    ));
    assert!(matches!(s.admit("r", || Ok(())), Err(Error::Poisoned(_))));
    // Owners cannot take the memory back while the session retains it.
    let fail = *dst.reclaim().unwrap_err();
    let (dst, e) = (fail.buffer, fail.error);
    assert!(matches!(e, Error::Busy(_)));
    // A failed quiesce keeps everything retained.
    gate.fail_quiesce.store(true, Ordering::SeqCst);
    assert!(s.recover().is_err());
    assert_eq!(s.state(), SessionState::Poisoned);
    gate.fail_quiesce.store(false, Ordering::SeqCst);
    s.recover().unwrap();
    assert_eq!((s.state(), s.retained()), (SessionState::Idle, 0));
    assert!(dst.regions().iter().all(|r| r.phase() == Phase::Lent));
    dst.reclaim().unwrap();
    src.reclaim().unwrap();
}

#[test]
fn failed_admission_hook_and_failed_consumer_poison() {
    let s = host_session();
    let (src, dst) = (buffer(4, 1), buffer(4, 0));
    s.reserve("r", vec![dst.share()]).unwrap();
    s.transfer(
        "r",
        0,
        &src.share(),
        &[Selection::prefix(0..2, 2)],
        Direction::Write,
    )
    .unwrap();
    s.fence("r").unwrap();
    assert!(s
        .admit("r", || Err(Error::Invalid("decoder refused".into())))
        .is_err());
    assert_eq!(s.state(), SessionState::Poisoned);
    s.recover().unwrap();

    s.reserve("q", vec![dst.share()]).unwrap();
    s.transfer(
        "q",
        0,
        &src.share(),
        &[Selection::prefix(0..2, 2)],
        Direction::Write,
    )
    .unwrap();
    s.fence("q").unwrap();
    s.admit("q", || Ok(())).unwrap();
    assert!(s.finish("q", false).is_err());
    assert_eq!(s.state(), SessionState::Poisoned);
    assert_eq!(s.retained(), 2);
}

#[test]
fn cancel_at_each_step() {
    let gate = Gate::new();
    let s = gate_session(&gate);
    let (src, dst) = (buffer(4, 1), buffer(4, 0));
    // Before any transfer: released immediately.
    s.reserve("a", vec![dst.share()]).unwrap();
    s.cancel("a").unwrap();
    assert_eq!(s.state(), SessionState::Idle);
    // In flight: fence still required, then admission discards.
    s.reserve("b", vec![dst.share()]).unwrap();
    s.transfer(
        "b",
        0,
        &src.share(),
        &[Selection::prefix(0..2, 4)],
        Direction::Write,
    )
    .unwrap();
    s.cancel("b").unwrap();
    assert_eq!(s.state(), SessionState::Transferring);
    assert!(s
        .transfer(
            "b",
            0,
            &src.share(),
            &[Selection::prefix(0..1, 1)],
            Direction::Write
        )
        .is_err());
    gate.open.store(true, Ordering::SeqCst);
    s.fence("b").unwrap();
    assert!(matches!(
        s.admit("b", || panic!("must not admit")),
        Err(Error::Cancelled(_))
    ));
    assert_eq!(s.state(), SessionState::Idle);
    // Admitted: must finish instead.
    s.reserve("c", vec![dst.share()]).unwrap();
    s.fence("c").unwrap();
    s.admit("c", || Ok(())).unwrap();
    assert!(matches!(s.cancel("c"), Err(Error::State(_))));
    assert!(matches!(s.cancel("other"), Err(Error::State(_))));
    s.finish("c", true).unwrap();
    // The flag does not leak into the next request.
    s.reserve("d", vec![dst.share()]).unwrap();
    s.transfer(
        "d",
        0,
        &src.share(),
        &[Selection::prefix(0..2, 1)],
        Direction::Write,
    )
    .unwrap();
    s.fence("d").unwrap();
    s.admit("d", || Ok(())).unwrap();
    s.finish("d", true).unwrap();
}

#[test]
fn single_flight_stale_ids_and_free_rejections() {
    let s = host_session();
    let (src, dst) = (buffer(4, 1), buffer(4, 0));
    s.reserve("r", vec![dst.share()]).unwrap();
    assert!(matches!(
        s.reserve("x", vec![dst.share()]),
        Err(Error::Busy(_))
    ));
    assert!(matches!(s.fence("x"), Err(Error::State(_))));
    assert!(s.reserve("", vec![dst.share()]).is_err());
    // Out-of-range selection, unknown group: rejected, nothing changes.
    assert!(s
        .transfer(
            "r",
            0,
            &src.share(),
            &[Selection::prefix(0..2, 9)],
            Direction::Write
        )
        .unwrap_err()
        .is_rejection());
    assert!(s
        .transfer(
            "r",
            1,
            &src.share(),
            &[Selection::prefix(0..2, 1)],
            Direction::Write
        )
        .unwrap_err()
        .is_rejection());
    assert_eq!(s.state(), SessionState::Reserved);
    // A destination in another memory kind or representation does not fit the route.
    let fp16 = Geometry {
        representation: Representation::Fp16,
        ..*spec(4).layout.geometry()
    };
    let other = KvSpec::new("m", "r", Layout::bhsd(fp16).unwrap()).unwrap();
    let n = other.layout.region_bytes() as usize;
    let regions = (0..4).map(|_| Region::host(n).unwrap()).collect();
    let fp16_buf = KvBuffer::new(other, regions).unwrap();
    let s2 = host_session();
    assert!(matches!(
        s2.reserve("r", vec![fp16_buf.share()]),
        Err(Error::Unsupported(_))
    ));
    s.cancel("r").unwrap();
}

#[test]
fn empty_suffix_fences_without_transfers() {
    let s = host_session();
    let dst = buffer(4, 0);
    s.reserve("r", vec![dst.share()]).unwrap();
    let r = s.fence("r").unwrap();
    assert_eq!((r.transfers, r.bytes), (0, 0));
    s.admit("r", || Ok(())).unwrap();
    s.finish("r", true).unwrap();
}

#[test]
fn per_layer_completion() {
    let gate = Gate::new();
    let s = gate_session(&gate);
    let (src, dst) = (buffer(4, 1), buffer(4, 0));
    s.reserve("r", vec![dst.share()]).unwrap();
    // Layer streaming: one transfer per layer (vLLM save_kv_layer).
    let t0 = s.transfer(
        "r",
        0,
        &src.share(),
        &[Selection::prefix(0..1, 4)],
        Direction::Write,
    );
    // A region can only carry one copy at a time: layer 1 lives in other regions (bhsd split).
    let t1 = s.transfer(
        "r",
        0,
        &src.share(),
        &[Selection::prefix(1..2, 4)],
        Direction::Write,
    );
    let (t0, _t1) = (t0.unwrap(), t1.unwrap());
    assert_eq!(gate.submits.load(Ordering::SeqCst), 2);
    assert_eq!(s.ready_layers("r").unwrap(), Vec::<u32>::new());
    assert!(!s.poll("r", t0).unwrap());
    gate.open.store(true, Ordering::SeqCst);
    assert!(s.poll("r", t0).unwrap());
    s.wait_layer("r", 1).unwrap();
    assert_eq!(s.ready_layers("r").unwrap(), vec![0, 1]);
    s.fence("r").unwrap();
    s.admit("r", || Ok(())).unwrap();
    s.finish("r", true).unwrap();
}

#[test]
fn pull_and_push_through_serialized_descriptors() {
    // Prefill and decode agents in one process; descriptors cross as bytes.
    let prefill = Agent::new("prefill", Some("prefill.local:5555".into()));
    let decode = Agent::new("decode", None);
    let meta = AgentMeta::from_bytes(&prefill.meta().to_bytes()).unwrap();
    assert_eq!(meta.agent_id, prefill.id());

    let src = buffer(8, 3);
    let wire = prefill
        .export(&src.share(), BTreeMap::new())
        .unwrap()
        .to_bytes();

    // Pull (vLLM NixlConnector style): the decode side reads the remote source.
    let dst = buffer(8, 0);
    let s = host_session();
    s.reserve("pull", vec![dst.share()]).unwrap();
    let remote_src = decode
        .resolve(&KvDescriptor::from_bytes(&wire).unwrap())
        .unwrap();
    s.transfer(
        "pull",
        0,
        &remote_src,
        &[Selection::prefix(0..2, 8)],
        Direction::Read,
    )
    .unwrap();
    s.fence("pull").unwrap();
    s.admit("pull", || Ok(())).unwrap();
    s.finish("pull", true).unwrap();
    drop(remote_src);

    // Push: the prefill side writes into a destination the decode side exported.
    let dst2 = buffer(8, 0);
    let dst_wire = decode
        .export(&dst2.share(), BTreeMap::new())
        .unwrap()
        .to_bytes();
    let remote_dst = prefill
        .resolve(&KvDescriptor::from_bytes(&dst_wire).unwrap())
        .unwrap();
    let p = host_session();
    p.reserve("push", vec![remote_dst]).unwrap();
    p.transfer(
        "push",
        0,
        &src.share(),
        &[Selection::prefix(0..2, 8)],
        Direction::Write,
    )
    .unwrap();
    p.fence("push").unwrap();
    p.admit("push", || Ok(())).unwrap();
    p.finish("push", true).unwrap();

    // Registered memory stays pinned until released.
    let dst2 = dst2.reclaim().unwrap_err().buffer;
    decode
        .release(&KvDescriptor::from_bytes(&dst_wire).unwrap())
        .unwrap();
    let a = dst.reclaim().unwrap();
    let b = dst2.reclaim().unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.as_slice(), y.as_slice());
    }
}

#[test]
fn several_selections_form_one_job() {
    // Two batch entries (blocks) of the same regions in one engine job: separate transfers would
    // collide on the single-copy-in-flight rule.
    let g = Geometry {
        representation: Representation::Bf16,
        kv: KvKind::KeyValue,
        layers: 2,
        blocks: 2,
        block_size: 8,
        heads: 2,
        head_dim: 4,
    };
    let spec = KvSpec::new("m", "r", Layout::bhsd(g).unwrap()).unwrap();
    let alloc = || {
        let n = spec.layout.region_bytes() as usize;
        KvBuffer::new(
            spec.clone(),
            (0..4).map(|_| Region::host(n).unwrap()).collect(),
        )
        .unwrap()
    };
    let (src, dst) = (alloc(), alloc());
    let s = host_session();
    s.reserve("r", vec![dst.share()]).unwrap();
    let sel = |b| Selection {
        tokens: Some(3),
        ..Selection::blocks(0..2, vec![b], vec![b])
    };
    s.transfer("r", 0, &src.share(), &[sel(0), sel(1)], Direction::Write)
        .unwrap();
    let r = s.fence("r").unwrap();
    assert_eq!((r.transfers, r.bytes), (1, 2 * 2 * 2 * 2 * 3 * 8));
    s.admit("r", || Ok(())).unwrap();
    s.finish("r", true).unwrap();
}
