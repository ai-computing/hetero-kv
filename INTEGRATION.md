# Integration guide

How each consumer's concepts map onto hetero-kv, and what is missing. "P0" marks what works in
this release; everything else names the phase that adds it (Issue #1).

## Shared vocabulary

| hetero-kv | Meaning |
|---|---|
| `Agent` / `AgentMeta` | A process' or engine instance's identity: registry id, name, opaque endpoint. Serializable handshake bytes. |
| `Region` / `RegionDesc` | A registered byte range in one memory kind (tier). Descriptor = what a peer needs to address it. |
| `KvSpec` / `Layout` | Model identity + physical layout (blocks, block size, heads, head_dim, representation, K/V or MLA latent, region split, strides). |
| `KvBuffer` / `KvDescriptor` | A spec plus its regions; the descriptor adds optional per-block content hashes. |
| `Selection` | Layers + token range through block tables (`blocks(src_ids, dst_ids)`, `prefix(n)`, `contiguous(...)`). |
| `Session` | One destination lease: reserve -> transfer (async, per layer if wanted) -> fence -> admit -> finish; cancel; poison/recover. |
| `Direction` | `write` (push, producer-initiated) or `read` (pull, consumer-initiated). |
| `KvEvent` / `EventLog` | BlockStored / BlockRemoved / BlockMoved / AllBlocksCleared, renderable in vLLM `KVEventBatch` array form. |
| Memory kinds | `host`, `host_pinned`, `cuda`, `rngd`, `atom`, `storage` (placeholder), `remote` (placeholder). |

## Optimus Prime (aicomp optimus-rm `accel/pd`) — P0, real

| Optimus | hetero-kv |
|---|---|
| `contracts.KVSpec(model_id, revision, representation, layout, shape)` | `KVSpec(model_id, revision, Layout)`; Optimus keeps its own spec for lane negotiation, the adapter derives the hetero-kv spec from the cache tensors |
| `Transport.supports(src, dst, spec)` | adapter checks the configured endpoints; engine routes are checked by `Session(...)` (`UnsupportedError`, no fallback) |
| `Transport.transfer(...)` "returns only after verified completion and visibility" | `reserve` + `transfer` + `fence` + `admit(notify)` (`hetero_kv.transfer_and_admit`) |
| `Transport.finish(request_id)` | `Session.finish(request_id)` |
| `recover()` | `Session.recover()` (engine quiesce); a still-admitted request is finished as failed first |
| `RetryableError` vs poison | hetero-kv rejections (`InvalidError`, `UnsupportedError`, `StateError`) guarantee nothing was submitted; the adapter re-raises them (`ValueError` subclasses) and Optimus' runtime keeps its own policy |
| `torch_cache.TorchCacheTransport` | `hetero_kv_optimus.TorchCacheTransport` (`TorchCopyEngine`, bhsd layout, one KV group per distinct layer shape) |
| `layer_progress.LayerProgress` | `hetero_kv.LayerProgress` (same semantics and messages) |
| `hetero_p2p.HeteroP2PTransport` + `CopySession` | `hetero_kv_optimus.SessionTransport` + `Session` (fenced = CopySession "ready", admitted = "decoding") |
| `npu/shared_kv.SharedKVSlab` (memfd host slab) | stays in Optimus; its mapping can be wrapped with `Region.from_buffer` for host-engine copies. Fd passing across processes is P3. |

Verified locally against optimus-rm `fff45de`: Optimus' own `tests/ -k pd` suite (193 passed,
21 skipped) gives the same result with `torch_cache.TorchCacheTransport` and `LayerProgress`
swapped for the adapter's.

## vLLM KV connector

vLLM V1 `KVConnectorBase_V1` (names as of v0.10/v0.11):

| vLLM hook | hetero-kv |
|---|---|
| `register_kv_caches(kv_caches)` | one `KVBuffer` over the per-layer tensors (`Layout.preset("layer_separate", ...)`, `tensor_region`), `Agent.export` -> descriptor |
| `request_finished(request, block_ids)` -> `kv_transfer_params` | `remote_engine_id` = agent id, `remote_block_ids` = block ids, `remote_host`/`remote_port` = `AgentMeta.endpoint`, plus the descriptor bytes |
| `get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta` | connector-side bookkeeping; hetero-kv not involved |
| `start_load_kv` | `reserve` local blocks; `transfer(Selection.blocks(remote_ids, local_ids, layers=(l, l+1)), direction="read")` per layer (pull, like NixlConnector) |
| `wait_for_layer_load(layer)` | `Session.wait_layer(request_id, layer)` |
| `save_kv_layer(layer, ...)` / `wait_for_save()` | push mode: `transfer(..., direction="write")` per layer / `fence` |
| `get_finished(...)` | `ready_layers` / `fence`, then `admit`; `finish` when the request ends |

Gaps: cross-process descriptor resolution and completion notification to the remote side (P3);
a CUDA engine for GPU tensors (P1; `TorchCopyEngine` works only within one process); MLA and
hybrid (multi-group) layouts are expressible but untested against vLLM's tensors.

Skeleton: `adapters/vllm/hetero_kv_vllm.py`.

## llm-d

| llm-d | hetero-kv |
|---|---|
| P/D routing sidecar passes `kv_transfer_params` between prefill and decode | carries hetero-kv `AgentMeta` and `KvDescriptor` bytes (versioned JSON) alongside vLLM's fields |
| KV-cache-aware router consumes vLLM KV events (ZMQ, msgpack `KVEventBatch`) | `Agent.publish(event)` and `Agent.drain_events()` -> `[ts, [[tag, ...], ...], dp_rank]`, the array form msgspec produces; the publisher msgpack-encodes and sends it |
| block hashes | `BlockHash` = int or bytes, carried per block in `KvDescriptor.block_hashes` and in events |

Gaps: no ZMQ publisher or msgpack encoder in hetero-kv (deliberate: transport only); event field
order follows vLLM v0.10/v0.11 and must be checked against the pinned version; `BlockMoved` is
rendered as `BlockRemoved` + `BlockStored` because vLLM has no move event. hetero-kv does no
routing.

## KV managers: LMCache, Mooncake, Dynamo KVBM

| Concept | hetero-kv |
|---|---|
| LMCache storage backend (`contains`, `put`/`batched_submit_put_task`, `get_blocking`, prefetch) | the manager keeps its index; puts/gets become `Session` transfers between tier buffers with `Selection.blocks` |
| Mooncake transfer engine (`registerLocalMemory`, `submitTransfer` READ/WRITE batches, `getTransferStatus`) | `Registry.register` / descriptors; `Session.transfer(direction=...)`; `poll`/`wait` |
| Dynamo KVBM layouts (LayerSeparate, FullyContiguous) and tiers (G1 GPU, G2 host pinned, G3 disk, G4 remote) | `Layout.preset("layer_separate" / "block_major")`; memory kinds `cuda`/`rngd`/`atom`, `host_pinned`, `storage`, `remote` |
| Block events for routers | `EventLog`, `BlockMoved` for tier moves |

Gaps: no storage or remote engine (kinds exist, routes are `Unsupported`); `host_pinned` is a
label in P0 (pinning needs the CUDA engine, P1); no eviction or index (by design).

Skeleton: `adapters/kvstore/hetero_kv_kvstore.py`.

## NIXL

| NIXL | hetero-kv |
|---|---|
| `nixlAgent`, `getLocalMD` / `loadRemoteMD` | `Agent`, `AgentMeta` bytes |
| `registerMem` (DRAM_SEG, VRAM_SEG, ...) | `Region` + `Registry.register`; memory kinds |
| `createXferReq(NIXL_READ/WRITE, local, remote)` / `postXferReq` / `getXferStatus` | L2 `Route::submit` with `Direction` / `Transfer::poll` |
| backend plugin (`nixlBackendEngine`) | planned thin C++ shim over the C ABI, see `adapters/nixl/README.md` |

Gaps (P3): the C ABI does not yet expose registration, descriptors or raw L2 span submission;
no notifications; no cross-process resolution.
