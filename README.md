# hetero-kv

Layered KV-cache transport between heterogeneous accelerators (GPU, RNGD, ATOM, host): a Rust
core, a C ABI and Python bindings. It only moves KV bytes; engines (Optimus Prime, vLLM, SGLang,
TensorRT-LLM), KV managers (LMCache, Mooncake, Dynamo KVBM) and llm-d keep model execution,
scheduling and indexing.

**Status: P0, host only.** L1–L4 API, host reference engine, Python-provided engines (a Torch
`copy_` engine is included), C ABI, CPU CI, and a tested Optimus Prime adapter. CUDA (P1), RNGD
PDMA <-> DMA-BUF (P2), NIXL and cross-process sessions (P3) follow; see Issue #1.

## Layers

| Layer | Rust module | What it owns |
|---|---|---|
| L5 | `hetero-kv-py`, `hetero-kv-capi`, `adapters/` | Python package `hetero_kv`, C ABI (`include/hetero_kv.h`), consumer adapters |
| L4 | `session` | reserve -> transfer -> fence -> admit -> finish, cancel, poison/recover; agents |
| L3 | `kv` | `KvSpec`, paged/strided `Layout`, `Selection` -> byte-span `Plan`, `LayerProgress`, buffers, descriptors, block events |
| L2 | `engine` | `CopyEngine`, capability matrix (device patterns, declared conversions), explicit `Route` (no fallback), async `Transfer`, `HostEngine` |
| L1 | `memory` | `Region` at a `Location` (kind, device index), byte-range ownership state machine (Owned -> Lent -> InFlight -> Done / Uncertain), registration |

Rules and diagrams: [ARCHITECTURE.md](ARCHITECTURE.md). Consumer mapping and gaps:
[INTEGRATION.md](INTEGRATION.md).

## Quick start

### Python

```sh
python3 -m venv .venv && .venv/bin/pip install maturin pytest
(cd crates/hetero-kv-py && ../../.venv/bin/maturin develop)
```

```python
import hetero_kv as hk

layout = hk.Layout.preset("bhsd", "bf16", layers=2, block_size=16, heads=2, head_dim=8)
spec = hk.KVSpec("my-model", "rev-1", layout)
new = lambda: hk.KVBuffer(spec, [hk.Region.host(layout.region_bytes) for _ in range(layout.region_count)])
src, dst = new(), new()

route = hk.Route(hk.HostEngine(), "host", "host", "bf16")        # explicit route; no fallback
session = hk.Session(route)                                       # one per request; share the route
session.reserve("req-1", dst)
tid = session.transfer("req-1", src, hk.Selection.prefix(5))      # async; returns a transfer id
receipt = session.fence("req-1")                                  # every completion verified
session.admit("req-1", notify=lambda: print("cache visible"))     # only after the fence
session.finish("req-1")
```

Locations name devices: `Region.external("rngd:3", addr, nbytes, owner)`,
`hk.Route(engine, "rngd:3", "atom:2", "bf16", "dlfloat16")` (only if the engine declared that
conversion). Engines can be Python objects (`name`, `capabilities`, `submit`, `poll`, `wait`,
`quiesce`); `hetero_kv.torch_engine.TorchCopyEngine` copies tensor-backed regions
(`host`, `cuda:N`) with `Tensor.copy_`. Many sessions over one route may be in flight at once on
one KV pool as long as their byte ranges are disjoint.

### Rust

```rust
use std::sync::Arc;
use hetero_kv::engine::{Direction, HostEngine, Route};
use hetero_kv::kv::{Geometry, KvBuffer, KvKind, KvSpec, Layout, Representation, Selection};
use hetero_kv::memory::{MemoryKind, Region};
use hetero_kv::session::Session;

let g = Geometry { representation: Representation::Bf16, kv: KvKind::KeyValue,
                   layers: 2, blocks: 1, block_size: 16, heads: 2, head_dim: 8 };
let spec = KvSpec::new("my-model", "rev-1", Layout::bhsd(g)?)?;
let alloc = || -> hetero_kv::Result<KvBuffer> {
    let n = spec.layout.region_bytes() as usize;
    KvBuffer::new(spec.clone(), (0..spec.layout.region_count()).map(|_| Region::host(n)).collect::<Result<_, _>>()?)
};
let (src, dst) = (alloc()?, alloc()?);
let route = Route::select(Arc::new(HostEngine::new()), MemoryKind::Host, MemoryKind::Host, Representation::Bf16)?;
// Share `route` (cheap clone) across threads; create one Session per request.
let session = Session::new(route);
session.reserve("req-1", vec![dst.share()])?;
session.transfer("req-1", 0, &src.share(), &[Selection::prefix(0..2, 5)], Direction::Write)?;
session.fence("req-1")?;
session.admit("req-1", || Ok(()))?;
session.finish("req-1", true)?;
```

### C

```c
#include "hetero_kv.h"   /* link with -lhetero_kv (target/<profile>/libhetero_kv.so) */

hkv_geometry g = {HKV_BF16, 0, 2, 1, 16, 2, 8};
hkv_spec *spec = hkv_spec_new("my-model", "rev-1", HKV_LAYOUT_BHSD, &g);
hkv_engine *engine = hkv_host_engine_new();
hkv_location host = {HKV_MEM_HOST, 0, false, 0};
hkv_session *s = hkv_session_new(engine, host, host, HKV_BF16, HKV_BF16);   /* one per request */
/* buffers: hkv_region_host_new + hkv_buffer_new; then
   hkv_session_reserve -> hkv_session_transfer -> hkv_session_fence -> hkv_session_admit -> hkv_session_finish;
   descriptors: hkv_agent_export -> hkv_descriptor_to_bytes / _from_bytes -> hkv_descriptor_resolve */
```

Full example: [`tests/c/smoke.c`](tests/c/smoke.c). Every call returns `hkv_status`;
`hkv_last_error()` has the message.

## Repository

```
crates/hetero-kv        core (L1-L4); cargo features cuda, rngd, atom, nixl reserved (empty)
crates/hetero-kv-capi   C ABI cdylib/staticlib; header generated into include/hetero_kv.h
crates/hetero-kv-py     PyO3 module + pure-Python package hetero_kv (py.typed, stubs)
adapters/               optimus (real), vllm, kvstore, nixl (skeletons / notes)
tests/python, tests/c   Python tests, C smoke test
```

## Development

```sh
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude hetero-kv-py
(cd crates/hetero-kv-py && maturin develop) && pytest tests/python adapters/optimus/tests
tests/c/run.sh
scripts/gen-header.sh   # after changing the C ABI (needs cbindgen); CI checks the header is current
```

Contribution rules (including "documented interfaces only"): [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Not declared yet. The ai-computing organization has no single convention (repositories use
Apache-2.0, BSD-3-Clause, MIT or none), so the choice is left to the maintainers.
