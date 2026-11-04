"""Python API: regions, layouts, plans, sessions, descriptors, events, layer progress."""

import pytest

import hetero_kv as hk


def spec(cap=8, layers=2, heads=2, head_dim=4, preset="bhsd", repr_="bf16", blocks=1):
    layout = hk.Layout.preset(preset, repr_, layers, cap, heads, head_dim, blocks=blocks)
    return hk.KVSpec("model", "rev", layout)


def buffer(s, seed=0):
    regions = []
    for r in range(s.layout.region_count):
        region = hk.Region.host(s.layout.region_bytes)
        region.write(0, bytes((seed + r * 31 + i) % 256 for i in range(s.layout.region_bytes)))
        regions.append(region)
    return hk.KVBuffer(s, regions)


def test_region_read_write_and_move():
    r = hk.Region.host(32)
    assert r.read() == bytes(32) and r.kind == "host" and r.state == "owned"
    r.write(4, b"abcd")
    assert r.read(4, 4) == b"abcd"
    with pytest.raises(ValueError):
        r.write(30, b"abcd")
    s = hk.KVSpec("m", "r", hk.Layout("fp16", 1, 4, 1, 2))
    buf = hk.KVBuffer(s, [r])
    assert r.state == "moved"
    with pytest.raises(hk.StateError):
        r.read()
    (back,) = buf.reclaim()
    assert back.read(4, 4) == b"abcd"


def test_failed_buffer_construction_keeps_regions():
    s = spec()
    small = [hk.Region.host(8) for _ in range(s.layout.region_count)]
    with pytest.raises(hk.InvalidError):
        hk.KVBuffer(s, small)
    assert all(r.state == "owned" for r in small)


def test_from_buffer_wraps_python_memory():
    backing = bytearray(128)
    s = hk.KVSpec("m", "r", hk.Layout("fp16", 1, 4, 2, 4))
    src = buffer(s, seed=9)
    dst = hk.KVBuffer(s, [hk.Region.from_buffer(backing)])
    session = hk.Session(hk.HostEngine(), representation="fp16")
    hk.transfer_and_admit(session, "r", src, dst, hk.Selection.prefix(4))
    session.finish("r")
    (expected,) = src.reclaim()
    assert bytes(backing) == expected.read()
    with pytest.raises(ValueError):
        hk.Region.from_buffer(b"read-only bytes")


def test_layouts_and_plans():
    lay = hk.Layout("bf16", 2, 16, 4, 8, blocks=3, order=["layer", "kv", "block", "token", "head"],
                    split="layer")
    assert lay.region_count == 2 and lay.strides["head"] == 16 and lay.split == "layer"
    padded = hk.Layout("bf16", 1, 4, 1, 8, strides={"kv": 256, "token": 32}, region_bytes=512)
    assert padded.region_bytes == 512
    with pytest.raises(hk.InvalidError):
        hk.Layout("bf16", 1, 4, 1, 8, strides={"kv": 8, "token": 8}, region_bytes=512)
    with pytest.raises(hk.InvalidError):
        hk.Layout("int8", 1, 4, 1, 8)
    a = spec(cap=8, preset="bhsd")
    b = spec(cap=32, preset="bhsd")
    spans = hk.plan(a, b, hk.Selection.prefix(5))
    assert len(spans) == 2 * 2 * 2 and all(n == 5 * 8 for *_, n in spans)
    paged = spec(cap=4, preset="layer_separate", blocks=6)
    gathered = hk.plan(paged, b, hk.Selection(src_blocks=[5, 0], tokens=7))
    assert sum(n for *_, n in gathered) == 2 * 2 * 2 * 7 * 8
    with pytest.raises(hk.InvalidError):
        hk.plan(a, hk.KVSpec("other", "rev", a.layout), hk.Selection.prefix(1))


def test_host_session_round_trip_and_receipt():
    s_src, s_dst = spec(cap=8), spec(cap=16)
    src, dst = buffer(s_src, 1), buffer(s_dst)
    session = hk.Session(hk.HostEngine())
    seen = []
    receipt = hk.transfer_and_admit(session, "r1", src, dst, hk.Selection.prefix(8), lambda: seen.append(1))
    assert seen == [1] and session.state == "admitted"
    assert receipt["bytes"] == 2 * 2 * 2 * 8 * 8 and receipt["engine"] == "host"
    session.finish("r1")
    assert session.state == "idle"
    a, b = src.reclaim(), dst.reclaim()
    for x, y in zip(a, b):
        for head in range(2):
            assert x.read(head * 64, 64) == y.read(head * 128, 64)


def test_rejections_poison_nothing_and_route_has_no_fallback():
    with pytest.raises(hk.UnsupportedError):
        hk.Session(hk.HostEngine(), "host", "cuda")
    session = hk.Session(hk.HostEngine())
    src, dst = buffer(spec()), buffer(spec())
    session.reserve("r", dst)
    with pytest.raises(hk.BusyError):
        session.reserve("x", dst)
    with pytest.raises(hk.InvalidError):
        session.transfer("r", src, hk.Selection.prefix(99))
    with pytest.raises(hk.StateError):
        session.admit("r")
    with pytest.raises(hk.StateError):
        session.fence("stale")
    assert session.state == "reserved"
    session.cancel("r")
    assert session.state == "idle"


def test_descriptors_events_and_agents():
    prefill, decode = hk.Agent("prefill", "prefill:5555"), hk.Agent("decode")
    meta = hk.decode_agent_meta(prefill.meta())
    assert meta["agent_id"] == prefill.id and meta["endpoint"] == "prefill:5555"
    s = spec(cap=4, preset="layer_separate", blocks=4)
    src = buffer(s, 5)
    wire = prefill.export(src, {0: 11, 2: b"\x01\x02"}).to_bytes()
    desc = hk.KVDescriptor.from_bytes(wire)
    assert desc.block_hashes == {0: 11, 2: b"\x01\x02"} and desc.agent_id == prefill.id
    with pytest.raises(hk.InvalidError):
        hk.KVDescriptor.from_bytes(b'{"hkv": 1, "type": "agent", "body": {}}')
    # Pull block 2 of the remote source into local block 0 (vLLM remote_block_ids style).
    dst = buffer(spec(cap=4, preset="block_major", blocks=2))
    session = hk.Session(hk.HostEngine())
    session.reserve("pull", dst)
    session.transfer("pull", decode.resolve(desc), hk.Selection.blocks([2], [0]), direction="read")
    session.fence("pull")
    session.admit("pull")
    session.finish("pull")
    with pytest.raises(hk.BusyError):
        src.reclaim()                 # still registered
    prefill.release(desc)
    src.reclaim()

    decode.publish({"type": "BlockStored", "block_hashes": [11], "parent_block_hash": None,
                    "token_ids": [1, 2, 3, 4], "block_size": 4, "lora_id": None, "medium": "CPU"})
    decode.publish({"type": "AllBlocksCleared"})
    ts, events, rank = decode.drain_events(data_parallel_rank=0)
    assert rank == 0 and isinstance(ts, float)
    assert events == [["BlockStored", [11], None, [1, 2, 3, 4], 4, None, "CPU"], ["AllBlocksCleared"]]
    assert decode.drain_events() is None
    with pytest.raises(hk.InvalidError):
        decode.publish({"type": "Nope"})


# Layer progress: the Optimus tests (test_pd_layer_progress.py), unchanged in substance.

def test_partial_prefix_and_out_of_order_layers_require_complete_coverage():
    p = hk.LayerProgress(3, 5, 9)
    p.publish(1, 3, 5, 9)
    p.publish(0, 1, 7, 9)
    assert p.drain() == [(1, 3, 5, 9), (0, 1, 7, 9)]
    with pytest.raises(ValueError, match="incomplete"):
        p.finish()
    p.publish(0, 1, 5, 7)
    p.finish()
    assert p.drain() == [(0, 1, 5, 7)]
    assert p.drain() == []
    with pytest.raises(ValueError, match="closed"):
        p.publish(0, 1, 5, 7)


def test_overlap_rejection_is_atomic_across_layers():
    p = hk.LayerProgress(3, 0, 4)
    p.publish(1, 2, 0, 2)
    with pytest.raises(ValueError, match="overlap"):
        p.publish(0, 3, 1, 3)
    p.publish(0, 1, 0, 4)
    p.publish(1, 2, 2, 4)
    p.publish(2, 3, 0, 4)
    p.finish()
    assert len(p.drain()) == 4


@pytest.mark.parametrize("span", [(-1, 1, 0, 4), (0, 4, 0, 4), (0, 1, -1, 4), (0, 1, 0, 5), (0, 0, 0, 4),
                                  (0, 1, 2, 2), (True, 2, 0, 4), (0, 1, 0.0, 4)])
def test_invalid_progress_never_enters_queue(span):
    p = hk.LayerProgress(3, 0, 4)
    with pytest.raises(ValueError):
        p.publish(*span)
    assert p.drain() == []


def test_empty_uncached_suffix_needs_no_transfer():
    p = hk.LayerProgress(3, 8, 8)
    p.finish()
    assert p.drain() == []
    assert p.finished and not p.pending
    with pytest.raises(ValueError, match="geometry"):
        hk.LayerProgress(0, 0, 1)
    with pytest.raises(ValueError, match="geometry"):
        hk.LayerProgress(1.0, 0, 1)


def test_layer_streaming_through_progress():
    """Producer publishes spans into a slab; consumer drains and copies each span as it arrives."""
    slab = hk.KVSpec("model", "rev", hk.Layout("bf16", 3, 12, 2, 4, order=["kv", "layer", "block", "head", "token"]))
    dest = spec(cap=20, layers=3)
    src, dst = buffer(slab, 3), buffer(dest)
    progress = hk.LayerProgress(3, 4, 12)   # reservation covers tokens 4..12; slab stores from 0
    session = hk.Session(hk.HostEngine())
    session.reserve("s", dst)
    for span in [(0, 2, 4, 8), (2, 3, 4, 12), (0, 2, 8, 12)]:
        progress.publish(*span)
        for l0, l1, t0, t1 in progress.drain():
            sel = hk.Selection.contiguous(t0 - 4, t0, t1 - t0, layers=(l0, l1))
            tid = session.transfer("s", src, sel)
            session.wait("s", tid)       # each span's regions are free again before the next one
    progress.finish()
    assert session.ready_layers("s") == [0, 1, 2]
    assert session.fence("s")["transfers"] == 3
    session.admit("s")
    session.finish("s")
    # Spot-check one row: layer 2, K plane, head 1, token 5 (slab index 1).
    (slab_region,) = src.reclaim()
    rows = dst.reclaim()
    row = 8
    slab_off = ((0 * 3 + 2) * 2 * 12 + 1 * 12 + 1) * row
    dst_off = (1 * 20 + 5) * row
    assert slab_region.read(slab_off, row) == rows[2 * 2 + 0].read(dst_off, row)
