"""The Optimus adapter keeps Optimus' Transport / LayerProgress behaviour.

Stand-ins below mirror the parts of ``opt_prime.accel.pd.contracts`` and ``opt_prime.kv_cache``
the transports touch, so this runs without aicomp. ``test_real_optimus_runtime`` additionally
drives Optimus' own PDRuntime when ``opt_prime`` is importable.
"""
import ctypes
from dataclasses import dataclass
from typing import Any

import pytest

import hetero_kv as hk
from hetero_kv_optimus import LayerProgress, SessionTransport, TorchCacheTransport, TransferPlan


@dataclass(frozen=True)
class Spec:                       # opt_prime.accel.pd.contracts.KVSpec (fields used here)
    model_id: str
    revision: str
    representation: str
    layout: str
    shape: tuple


@dataclass(frozen=True)
class Lease:                      # opt_prime.accel.pd.contracts.KVLease (fields used here)
    spec: Spec
    owner: Any
    payload: Any
    endpoint: str


def test_layer_progress_is_the_library_type():
    assert LayerProgress is hk.LayerProgress
    p = LayerProgress(2, 0, 4)
    p.publish(0, 2, 0, 4)
    p.finish()


# --- hetero_p2p bridge replacement --------------------------------------------------------

SPEC = Spec("test-model", "immutable-revision", "fp16", "linear-v1", (4,))


class Engine:
    """CPU engine recording the order of fence (completion) and notify."""

    name = "fake-dma"
    capabilities = [("host", "host", "fp16")]

    def __init__(self, events, fail=False):
        self.events, self.fail, self.jobs = events, fail, []

    def submit(self, job):
        self.jobs.append(job)
        return job

    def poll(self, job):
        return False

    def wait(self, job):
        if self.fail:
            raise RuntimeError("DMA completion unknown")
        for si, so, di, do, n in job.spans:
            ctypes.memmove(job.dst[di].addr + do, job.src[si].addr + so, n)
        self.events.append("fence")
        return job.nbytes

    def quiesce(self):
        return True


def hk_buffer(seed=0, representation="fp16"):
    spec = hk.KVSpec(SPEC.model_id, SPEC.revision, hk.Layout(representation, 2, 8, 1, 4))
    region = hk.Region.host(spec.layout.region_bytes)
    region.write(0, bytes((seed + i) % 256 for i in range(spec.layout.region_bytes)))
    return hk.KVBuffer(spec, [region])


def bridge(events, fail=False, plan=None):
    src, dst = hk_buffer(3), hk_buffer()
    plan = plan or TransferPlan(src, [dst], [[hk.Selection.prefix(8)]])
    return SessionTransport(Engine(events, fail), lambda s, d: plan, lambda d: events.append("notify"),
                            source="rngd:3", destination="atom:2", spec=SPEC), src, dst


def test_p2p_bridge_fence_then_notify_then_finish():
    events = []
    t, src, dst = bridge(events)
    source, destination = Lease(SPEC, t, None, "rngd:3"), Lease(SPEC, t, None, "atom:2")
    receipt = t.transfer("x", source, destination)
    assert events == ["fence", "notify"] and receipt["bytes"] == 2 * 2 * 8 * 4 * 2
    assert t.state == "admitted"                 # CopySession "decoding"
    t.finish("x")
    assert t.state == "idle"
    (a,), (b,) = src.reclaim(), dst.reclaim()
    assert a.read() == b.read()


def test_p2p_bridge_rejects_route_and_format_before_copying():
    events = []
    t, _, _ = bridge(events)
    other = Lease(SPEC, t, None, "rngd:0")
    with pytest.raises(ValueError, match="unsupported P2P route"):
        t.transfer("x", other, Lease(SPEC, t, None, "atom:2"))
    bf16_plan = TransferPlan(hk_buffer(representation="bf16"), [hk_buffer()], [[hk.Selection.prefix(8)]])
    t2, _, _ = bridge(events, plan=bf16_plan)
    with pytest.raises(ValueError, match="format differs"):
        t2.transfer("x", Lease(SPEC, t, None, "rngd:3"), Lease(SPEC, t, None, "atom:2"))
    assert events == [] and t.state == "idle" and t2.state == "idle"


def test_p2p_bridge_uncertain_completion_poisons_until_recovered():
    events = []
    t, _, dst = bridge(events, fail=True)
    with pytest.raises(hk.TransferError):
        t.transfer("x", Lease(SPEC, t, None, "rngd:3"), Lease(SPEC, t, None, "atom:2"))
    assert t.state == "poisoned" and "notify" not in events
    with pytest.raises(hk.BusyError):
        dst.reclaim()
    assert t.recover() is True and t.state == "idle"


# --- torch_cache replacement --------------------------------------------------------------

torch = pytest.importorskip("torch")


class KVCacheManager:
    """Storage layout and API of opt_prime.kv_cache.KVCacheManager used by the transport."""

    def __init__(self, layers, batch, max_seq_len, heads, head_dim, dtype, device="cpu"):
        self.layer_start, self.layer_end, self.max_seq_len = 0, layers, max_seq_len
        self.device, self.dtype, self._seq_len = torch.device(device), dtype, 0
        self._cache = {i: (torch.zeros(batch, heads, max_seq_len, head_dim, dtype=dtype, device=device),
                           torch.zeros(batch, heads, max_seq_len, head_dim, dtype=dtype, device=device))
                       for i in range(layers)}

    def get(self, i):
        k, v = self._cache[i]
        return k[:, :, :self._seq_len], v[:, :, :self._seq_len]

    def get_seq_len(self):
        return self._seq_len

    def set_seq_len(self, n):
        self._seq_len = n


class LayeredKVCache:
    """opt_prime.accel.pd.torch_cache.LayeredKVCache: per-layer K/V shapes."""

    def __init__(self, layer_dims, max_seq_len, dtype):
        self.layer_start, self.layer_end, self.max_seq_len = 0, len(layer_dims), max_seq_len
        self.k = [torch.zeros(1, h, max_seq_len, d, dtype=dtype) for h, d in layer_dims]
        self.v = [torch.zeros(1, h, max_seq_len, d, dtype=dtype) for h, d in layer_dims]
        self._len = 0

    def get(self, i):
        return self.k[i][:, :, :self._len], self.v[i][:, :, :self._len]

    def get_seq_len(self):
        return self._len

    def set_seq_len(self, n):
        self._len = n


@dataclass
class TorchCache:
    manager: Any


def reference_bytes(src):
    """What Optimus' TorchCacheTransport reports: bytes of every get(i) tensor."""
    total = 0
    for i in range(src.layer_start, src.layer_end):
        k, v = src.get(i)
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def fill(manager, n):
    gen = torch.Generator().manual_seed(n)
    pairs = [manager._cache[i] for i in range(manager.layer_end)] if hasattr(manager, "_cache") else \
        list(zip(manager.k, manager.v))
    for k, v in pairs:
        k[:, :, :n] = torch.randn(k[:, :, :n].shape, generator=gen).to(k.dtype)
        v[:, :, :n] = torch.randn(v[:, :, :n].shape, generator=gen).to(v.dtype)
    manager.set_seq_len(n)


@pytest.mark.parametrize("make", [
    lambda cap: KVCacheManager(3, 1, cap, 2, 8, torch.float32),
    lambda cap: KVCacheManager(2, 2, cap, 2, 4, torch.bfloat16),       # batch > 1
    lambda cap: LayeredKVCache([(2, 8), (1, 16), (2, 8)], cap, torch.float32),
])
def test_torch_cache_transport_matches_optimus(make):
    spec = Spec("tiny", "seed17", "torch.float32", "torch-bhsd-full-attention-v1:test", (1,))
    src, dst = make(16), make(16)
    fill(src, 7)
    transport = TorchCacheTransport(source="source", destination="destination", spec=spec)
    source = Lease(spec, src, TorchCache(src), "source")
    destination = Lease(spec, dst, TorchCache(dst), "destination")
    assert transport.supports("source", "destination", spec)
    assert not transport.supports("destination", "source", spec)
    receipt = transport.transfer("r", source, destination)
    assert receipt == dict(transport="torch-copy", bytes=reference_bytes(src), cache_position=7, completed=True)
    assert dst.get_seq_len() == 7
    for i in range(src.layer_end):
        for a, b in zip(src.get(i), dst.get(i)):
            torch.testing.assert_close(b, a, rtol=0, atol=0)
    transport.finish("r")
    # The lane is reusable, as before.
    fill(src, 3)
    assert transport.transfer("r2", source, destination)["cache_position"] == 3
    transport.finish("r2")
    assert transport.recover() is True


def test_torch_cache_transport_rejections_leave_lane_usable():
    spec = Spec("tiny", "seed17", "torch.float32", "l", (1,))
    src, dst = KVCacheManager(1, 1, 8, 1, 4, torch.float32), KVCacheManager(1, 1, 8, 1, 4, torch.float32)
    t = TorchCacheTransport(source="source", destination="destination", spec=spec)
    source, destination = Lease(spec, src, TorchCache(src), "source"), Lease(spec, dst, TorchCache(dst), "destination")
    with pytest.raises(ValueError, match="invalid source sequence length"):
        t.transfer("r", source, destination)
    with pytest.raises(ValueError, match="unconfigured tensor-copy route"):
        t.transfer("r", source, Lease(spec, dst, TorchCache(dst), "elsewhere"))
    fill(src, 8)
    t.transfer("r", source, destination)
    # Decoder failed: Optimus poisons the lane and calls recover(); the session settles.
    assert t.recover() is True
    t.transfer("r3", source, destination)
    t.finish("r3")


def test_real_optimus_runtime():
    """Optimus' PDRuntime + TorchCausalLMEngine with this transport (needs opt_prime + transformers)."""
    pd = pytest.importorskip("opt_prime.accel.pd")
    transformers = pytest.importorskip("transformers")
    from opt_prime.accel.pd.torch_engine import TorchCausalLMEngine

    torch.manual_seed(17)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=41, max_position_embeddings=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, head_dim=8, num_attention_heads=4, num_key_value_heads=2)).eval()
    import copy
    for prefix in (False, True):
        source = TorchCausalLMEngine(copy.deepcopy(model), model_id="llama", revision="seed17", max_context=128,
                                     endpoint="source", prefix_only=prefix)
        dest = TorchCausalLMEngine(copy.deepcopy(model), model_id="llama", revision="seed17", max_context=128,
                                   endpoint="destination")
        runtime = pd.PDRuntime(source, dest, TorchCacheTransport(source="source", destination="destination",
                                                                 spec=source.spec))
        for length, count in ((2, 1), (7, 6), (19, 16)):
            prompt = tuple(1 + i % 39 for i in range(length))
            full = list(prompt)
            with torch.inference_mode():
                for _ in range(count):
                    full.append(int(model(torch.tensor([full]), use_cache=False).logits[0, -1].argmax()))
            assert runtime.generate(pd.PDRequest(f"{length}-{count}", prompt, count)) == full[length:]
            assert runtime.state == "idle" and runtime.last_receipt["bytes"] > 0
