"""One shared Route per engine, one Session per request, many requests in flight on one pool;
device locations and declared conversions."""
import ctypes
import threading

import pytest

import hetero_kv as hk

N_BLOCKS, LAYERS = 8, 3


def pool_spec(blocks):
    return hk.KVSpec("m", "r", hk.Layout.preset("layer_separate", "fp16", LAYERS, 4, 2, 8, blocks=blocks))


def make(blocks, fill=None):
    s = pool_spec(blocks)
    regions = []
    for _ in range(s.layout.region_count):
        r = hk.Region.host(s.layout.region_bytes)
        if fill is not None:
            r.write(0, bytes([fill]) * s.layout.region_bytes)
        regions.append(r)
    return hk.KVBuffer(s, regions)


class Deferred:
    """Copies at wait(), so submitted jobs are genuinely in flight together."""

    name = "deferred"
    capabilities = [("host", "host", None)]

    def __init__(self):
        self.submitted = 0
        self.lock = threading.Lock()

    def submit(self, job):
        with self.lock:
            self.submitted += 1
        return job

    def poll(self, job):
        return False

    def wait(self, job):
        for si, so, di, do, n in job.spans:
            ctypes.memmove(job.dst[di].addr + do, job.src[si].addr + so, n)
        return job.nbytes

    def quiesce(self):
        return True


@pytest.mark.parametrize("engine", [hk.HostEngine(), Deferred()])
def test_many_sessions_one_route_one_pool(engine):
    route = hk.Route(engine, "host", "host", "fp16")
    pool = make(N_BLOCKS)
    sources = [make(1, fill=i + 1) for i in range(N_BLOCKS)]
    submitted = threading.Barrier(N_BLOCKS)
    errors = []

    def request(i):
        try:
            rid = f"req-{i}"
            s = hk.Session(route)                   # cheap: shares the route's engine
            s.reserve(rid, pool)
            s.transfer(rid, sources[i], hk.Selection.blocks([0], [i]))
            submitted.wait()                         # all requests in flight on the same pool
            s.fence(rid)
            s.admit(rid)
            s.finish(rid)
        except Exception as e:                       # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=request, args=(i,)) for i in range(N_BLOCKS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    block = 4 * 2 * 8 * 2                            # tokens x heads x head_dim x fp16
    for region in pool.reclaim():
        data = region.read()
        for i in range(N_BLOCKS):
            for kv in range(2):
                off = kv * N_BLOCKS * block + i * block
                assert data[off:off + block] == bytes([i + 1]) * block


def test_overlapping_block_is_busy_and_changes_nothing():
    engine = Deferred()
    route = hk.Route(engine, "host", "host", "fp16")
    pool, a, b = make(4), make(1, 1), make(1, 2)
    s1, s2 = hk.Session(route), hk.Session(route)
    s1.reserve("a", pool)
    s2.reserve("b", pool)
    s1.transfer("a", a, hk.Selection.blocks([0], [2]))
    with pytest.raises(hk.BusyError):
        s2.transfer("b", b, hk.Selection.blocks([0], [2]))
    assert engine.submitted == 1 and s2.state == "reserved"
    s2.transfer("b", b, hk.Selection.blocks([0], [3]))
    for s, rid in ((s1, "a"), (s2, "b")):
        s.fence(rid)
        s.admit(rid)
        s.finish(rid)
    assert pool.phases() == ["done"] * LAYERS


class DeviceEngine:
    """Declares cuda:0 -> any CUDA device, and an RNGD:3 -> ATOM BF16 -> DLFloat16 conversion."""

    name = "fake-devices"
    capabilities = [("cuda:0", "cuda", "bf16"), ("rngd:3", "atom", "bf16", "dlfloat16")]

    def __init__(self):
        self.jobs = []

    def submit(self, job):
        self.jobs.append(job)
        return job

    def poll(self, job):
        return True

    def wait(self, job):
        return job.nbytes

    def quiesce(self):
        return True


def device_buffer(location, representation):
    s = hk.KVSpec("m", "r", hk.Layout.preset("block_major", representation, 2, 4, 1, 8, blocks=2))
    return hk.KVBuffer(s, [hk.Region.external(location, 0x1000, s.layout.region_bytes, None)])


def test_device_locations_and_declared_conversion():
    engine = DeviceEngine()
    assert hk.Route(engine, "cuda:0", "cuda:1", "bf16").dst == "cuda:1"
    for args in [("cuda:1", "cuda:0", "bf16"), ("rngd:0", "atom:2", "bf16", "dlfloat16"),
                 ("rngd:3", "atom:2", "bf16"), ("cuda:0", "cuda:0@5", "bf16")]:
        with pytest.raises(hk.UnsupportedError):
            hk.Route(engine, *args)

    route = hk.Route(engine, "rngd:3", "atom:2", "bf16", "dlfloat16")
    src, dst = device_buffer("rngd:3", "bf16"), device_buffer("atom:2", "dlfloat16")
    assert src.location == "rngd:3"
    s = hk.Session(route)
    with pytest.raises(hk.UnsupportedError):
        s.reserve("r", device_buffer("atom:1", "dlfloat16"))   # another ATOM device
    s.reserve("r", dst)
    s.transfer("r", src, hk.Selection.blocks([0, 1], [1, 0]))
    s.fence("r")
    job = engine.jobs[-1]
    assert (job.representation, job.dst_representation) == ("bf16", "dlfloat16")
    assert {i.location for i in job.src} == {"rngd:3"} and job.dst[0].index == 2
    # Without the converting route, different representations are not planned.
    with pytest.raises(hk.InvalidError):
        hk.plan(src.spec, dst.spec, hk.Selection.blocks([0], [0]))
