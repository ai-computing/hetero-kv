"""Adapter skeletons stay importable; the parts that already work (events) behave."""
import importlib.util
from pathlib import Path

import pytest

import hetero_kv as hk

ROOT = Path(__file__).resolve().parents[2] / "adapters"


def load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vllm_skeleton_is_explicitly_unimplemented():
    m = load("vllm/hetero_kv_vllm.py")
    c = m.HeteroKVConnector("prefill-0", hk.HostEngine(), "bf16", "host", "host")
    with pytest.raises(NotImplementedError):
        c.wait_for_layer_load("model.layers.0.self_attn")


def test_kvstore_skeleton_publishes_router_events():
    m = load("kvstore/hetero_kv_kvstore.py")
    layout = hk.Layout.preset("block_major", "bf16", 2, 16, 1, 8, blocks=4)
    spec = hk.KVSpec("m", "r", layout)
    agent = hk.Agent("kvstore")
    store = m.TieredBlockStore(agent, {"host": hk.KVBuffer(spec, [hk.Region.host(layout.region_bytes)])}, {})
    store.put(7, "host", 0, list(range(16)))
    assert store.contains(7)
    store.remove(7)
    _, events, _ = agent.drain_events()
    assert [e[0] for e in events] == ["BlockStored", "BlockRemoved"]
    assert events[0][4] == 16
