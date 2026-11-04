"""Engines implemented in Python and called from the Rust core."""
import ctypes

import pytest

import hetero_kv as hk


def spec(cap=8):
    return hk.KVSpec("model", "rev", hk.Layout.preset("bhsd", "fp16", 2, cap, 2, 4))


def buffer(s, seed=0):
    regions = []
    for r in range(s.layout.region_count):
        region = hk.Region.host(s.layout.region_bytes)
        region.write(0, bytes((seed + r + i) % 251 for i in range(s.layout.region_bytes)))
        regions.append(region)
    return hk.KVBuffer(s, regions)


class MemmoveEngine:
    """Deferred host engine: copies happen in wait(), so completion is genuinely asynchronous."""

    name = "py-memmove"
    capabilities = [("host", "host", "fp16")]

    def __init__(self, report=None, fail=False, quiesce_ok=True):
        self.jobs, self.report, self.fail, self.quiesce_ok = [], report, fail, quiesce_ok
        self.quiesced = 0

    def submit(self, job):
        assert job.representation == "fp16" and job.direction in ("write", "read")
        self.jobs.append(job)
        return len(self.jobs) - 1

    def poll(self, ticket):
        return False

    def wait(self, ticket):
        if self.fail:
            raise RuntimeError("completion queue error")
        job = self.jobs[ticket]
        for si, so, di, do, n in job.spans:
            ctypes.memmove(job.dst[di].addr + do, job.src[si].addr + so, n)
        return job.nbytes if self.report is None else self.report

    def quiesce(self):
        self.quiesced += 1
        return self.quiesce_ok


def test_python_engine_round_trip():
    engine = MemmoveEngine()
    session = hk.Session(engine, representation="fp16")
    assert session.engine == "py-memmove"
    src, dst = buffer(spec(), 7), buffer(spec(16))
    session.reserve("r", dst)
    tid = session.transfer("r", src, hk.Selection.prefix(6))
    assert not session.poll("r", tid)
    assert dst.phases() == ["in_flight"] * 4
    receipt = session.fence("r")
    assert receipt["bytes"] == 2 * 2 * 2 * 6 * 8 and dst.phases() == ["done"] * 4
    session.admit("r")
    session.finish("r")
    for a, b in zip(src.reclaim(), dst.reclaim()):
        for head in range(2):
            assert a.read(head * 64, 48) == b.read(head * 128, 48)


def test_route_needs_advertised_representation():
    with pytest.raises(hk.UnsupportedError):
        hk.Session(MemmoveEngine(), representation="bf16")
    with pytest.raises(ValueError):
        hk.Session(object())


@pytest.mark.parametrize("engine", [MemmoveEngine(fail=True), MemmoveEngine(report=1)])
def test_engine_failure_or_short_completion_poisons(engine):
    session = hk.Session(engine, representation="fp16")
    src, dst = buffer(spec()), buffer(spec())
    session.reserve("r", dst)
    session.transfer("r", src, hk.Selection.prefix(8))
    with pytest.raises(hk.TransferError):
        session.fence("r")
    assert session.state == "poisoned" and session.retained == 2
    assert dst.phases() == ["uncertain"] * 4
    with pytest.raises(hk.PoisonedError):
        session.reserve("next", dst)
    with pytest.raises(hk.BusyError):
        dst.reclaim()
    session.recover()
    assert engine.quiesced == 1 and session.state == "idle"
    dst.reclaim()


def test_declined_quiesce_keeps_session_poisoned():
    engine = MemmoveEngine(fail=True, quiesce_ok=False)
    session = hk.Session(engine, representation="fp16")
    src, dst = buffer(spec()), buffer(spec())
    session.reserve("r", dst)
    session.transfer("r", src, hk.Selection.prefix(8))
    with pytest.raises(hk.TransferError):
        session.fence("r")
    with pytest.raises(hk.TransferError):
        session.recover()
    assert session.state == "poisoned"


def test_failing_admission_hook_poisons():
    session = hk.Session(MemmoveEngine(), representation="fp16")
    src, dst = buffer(spec()), buffer(spec())
    session.reserve("r", dst)
    session.transfer("r", src, hk.Selection.prefix(8))
    session.fence("r")

    def notify():
        raise RuntimeError("decoder refused")

    with pytest.raises(hk.TransferError, match="decoder refused"):
        session.admit("r", notify)
    assert session.state == "poisoned"


def test_cancel_after_submit_fences_then_discards():
    engine = MemmoveEngine()
    session = hk.Session(engine, representation="fp16")
    src, dst = buffer(spec()), buffer(spec())
    session.reserve("r", dst)
    session.transfer("r", src, hk.Selection.prefix(8))
    session.cancel("r")
    session.fence("r")
    with pytest.raises(hk.CancelledError):
        session.admit("r", lambda: pytest.fail("must not admit"))
    assert session.state == "idle"
    dst.reclaim()
