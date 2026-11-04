# hetero-kv-optimus

Drop-in replacements for the KV transport pieces of aicomp optimus-rm `opt_prime/accel/pd`.
Duck-typed against Optimus' contracts; nothing imports `opt_prime`.

| Optimus module | Replacement | Notes |
|---|---|---|
| `layer_progress.LayerProgress` | `hetero_kv_optimus.LayerProgress` | Same class as `hetero_kv.LayerProgress`; same messages and exact-`int` checks |
| `torch_cache.TorchCacheTransport` | `hetero_kv_optimus.TorchCacheTransport` | Same constructor and receipt; copies run through a `Session` with `TorchCopyEngine` |
| `hetero_p2p.HeteroP2PTransport` | `hetero_kv_optimus.SessionTransport` | Any hetero-kv engine; `plan_builder` returns a `TransferPlan` (buffers + selections) |

```python
# layer_progress.py
from hetero_kv_optimus import LayerProgress  # noqa: F401

# torch_cache.py (LayeredKVCache, TorchCache and fence stay in Optimus)
from hetero_kv_optimus import TorchCacheTransport  # noqa: F401
```

Install: `pip install ./crates/hetero-kv-py ./adapters/optimus` (or `maturin develop` in
`crates/hetero-kv-py` and `pip install -e adapters/optimus`).

Tests: `pytest adapters/optimus/tests`. `test_real_optimus_runtime` drives Optimus' own
`PDRuntime` + `TorchCausalLMEngine` when `opt_prime` and `transformers` are importable; it skips
in CI.
