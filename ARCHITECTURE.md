# Architecture

hetero-kv only moves KV bytes. Model execution, batching, scheduling, KV indexing and serving
stay in the engines and KV managers that use it. Design: Issue #1; P0 scope: Issue #2.

## Layers

```
 L5 bindings   hetero-kv-py (Python `hetero_kv`) · hetero-kv-capi (C ABI, include/hetero_kv.h)
               adapters/: optimus (real) · vllm, kvstore, nixl (skeletons / notes)
 L4 session    reserve -> transfer -> fence -> admit -> finish; cancel; poison      session::
               agents: identity + registry + events; handshake metadata
 L3 KV         KvSpec, Layout (blocks, strides, K/V or MLA latent), Selection -> Plan  kv::
               LayerProgress; KvBuffer / KvDescriptor (block hashes); KV events
 L2 engines    CopyEngine trait, Capabilities, Route (explicit, no fallback),         engine::
               Transfer handles; HostEngine; Python engines (via hetero-kv-py)
 L1 memory     Region (memory kinds/tiers), ownership state machine, Registry,        memory::
               RegionDesc
 L0 sys        driver UAPI / SDK bindings, one cargo feature per vendor (empty in P0)
 --            wire: versioned JSON envelope shared by L1/L3/L4 descriptors           wire::
```

Rules:

1. **One direction of dependency.** L4 uses L3, L3 uses L2, L2 uses L1; never upwards. Bindings
   and adapters sit on top and are never imported by the core. Vendor code (L0) is reached only
   through engines behind cargo features (`cuda`, `rngd`, `atom`, `nixl`; empty placeholders in
   P0).
2. **Decode is admitted only after every completion is fenced.** `Session::admit` runs the
   consumer's visibility hook only in state `Fenced`.
3. **Poison on uncertainty.** An engine error, a completion that reports a different byte count
   than planned, a dropped unfinished transfer handle, a failing admission hook or a failed
   consumer poisons the session. Its buffers are retained and their regions are `Uncertain`;
   nothing is reused until an engine quiesce succeeds (`Session::recover`). Memory that is still
   uncertain when its last reference drops is leaked, never freed.
4. **Explicit routes, no silent fallback.** A route is selected for `(source kind, destination
   kind, representation)`; if the engine does not advertise it, selection fails with
   `Unsupported`. The host engine never stands in for a device route.
5. **Rejections are free.** Invalid geometry, out-of-range selections, wrong lifecycle steps and
   stale request ids fail before anything is submitted and change no state.
6. **Only documented interfaces.** Engines may use kernel driver UAPI, public vendor SDK APIs,
   DMA-BUF and the CUDA driver API. No hooks into vendor binaries (no LD_PRELOAD shims, symbol
   interposition, binary patching or private ioctls reverse-engineered from closed libraries),
   and no dependence on patched or hash-pinned third-party binaries. This holds for the whole
   repository, including adapters, tests and CI.
7. **CPU-only CI.** The host engine and Python-provided engines cover L1-L4 without hardware.

## Ownership state machine (L1)

```
            lend()                 Route::submit             Transfer::wait ok
  Owned ───────────► Lent ───────────────────────► InFlight ─────────────────► Done
    ▲                 ▲                               │                         │
    │                 │ Session::recover (quiesced)   │ engine error, short     │ submit again
    │                 └──────────── Uncertain ◄───────┘ completion, handle      │ (Done -> InFlight)
    │                                  │                dropped unfinished      │
    │  reclaim_quiesced(&Quiesced)     │                                        │
    ├──────────────────────────────────┘                                        │
    │  reclaim()  (no RegionRef alive, phase Lent | Done)                       │
    └───────────────────────────────────────────────────────────────────────────┘
```

How the types enforce it:

| Rule | Mechanism |
|---|---|
| Only the owner touches host bytes | `Region` is unique (`!Clone`); `as_slice`/`as_mut_slice` exist only on `Region` |
| No reuse while a transfer may reference the memory | `Transfer`, sessions and registries hold counted `RegionRef`s; `Lent::reclaim` fails while any is alive |
| One copy in flight per region | `Route::submit` moves every touched region `Lent/Done -> InFlight` atomically (rolls back on rejection) |
| Uncertain needs an explicit quiesce | `reclaim_quiesced` and `RegionRef::settle` take a `Quiesced` proof that only `Route::quiesce` (engine quiesce succeeded) can create |
| Abandoned work is uncertain | `Drop for Transfer` marks unfinished transfers' regions `Uncertain` |

## Session lifecycle (L4)

```
  Idle ─reserve─► Reserved ─transfer─► Transferring ─fence─► Fenced ─admit─► Admitted ─finish(ok)─► Idle
                     │ cancel ► Idle        │ cancel: flag;          │ cancel ► Idle   │ finish(failed)
                     │                      │ fence, then admit      │                 ▼
                     │                      │ discards (Cancelled)   │              Poisoned ─recover─► Idle
                     └──────────────── any engine error / failed admission hook ──────┘
```

Single flight: one request per session; `reserve` on a busy session is `Busy`. A session may hold
several destination buffers (one per KV group, e.g. hybrid models whose layers have different
shapes). Transfers are asynchronous: `transfer` returns when the engine accepted the job;
`poll`/`wait` (per transfer), `wait_layer`/`ready_layers` (per layer) observe completion, which
is how vLLM's `wait_for_layer_load` and Optimus' layer streaming fit. Admission still needs
`fence`.

`Direction` (`Write` = push by the producer, `Read` = pull by the consumer) is carried to engines
and descriptors. P0 resolves descriptors between agents in one process; cross-process transport
is P3.

## KV layouts and plans (L3)

A `Layout` places each row (`head_dim` elements of one layer, plane, block, token, head) at a byte
offset in one of the buffer's regions:

- `Geometry`: representation (`bf16`, `fp16`, `dlfloat16`, `fp32`: distinct, never aliased),
  `KvKind` (K/V planes or one MLA latent plane), layers, blocks, block size, heads, head_dim.
- `Split`: one region, one per layer, or one per (layer, plane). Fused K/V = same region,
  separated by the `kv` stride.
- Strides: derived from an axis order (`Layout::contiguous`) or explicit (`Layout::strided`,
  validated for non-overlap). Presets: `layer_separate` (vLLM per-layer tensors, Dynamo
  LayerSeparate), `block_major` (Dynamo FullyContiguous, LMCache chunks), `layer_major`, `bhsd`
  (Hugging Face / Optimus `[batch, heads, seq, dim]` per layer and plane).

A `Selection` addresses tokens through block tables on each side; `plan()` emits coalesced byte
spans between any two compatible layouts (same model, revision, representation, planes, layers,
heads, head_dim). A contiguous cache is one block, so a token-major prefix copy is
`Selection::prefix`. `LayerProgress` validates producer write coverage (Optimus semantics).

## How engines plug in (L2)

```rust
pub trait CopyEngine: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> &Capabilities;                  // (src kind, dst kind, representation)
    fn submit(&self, job: &CopyJob<'_>) -> Result<Box<dyn Completion>>;   // must not block on the device
    fn quiesce(&self) -> Result<()>;                          // nothing accepted is still running
}
pub trait Completion: Send {
    fn poll(&mut self) -> Result<Option<u64>>;                // Some(bytes) once complete
    fn wait(&mut self) -> Result<u64>;                        // bytes verified as copied
}
```

`Route::submit` validates spans (bounds, non-overlapping destinations, kinds) before the engine
sees them, and compares the engine's reported byte count with the plan. Engines in this repo:

- `HostEngine`: synchronous `memmove` between host regions.
- Python engines (`hetero-kv-py`): any object with `name`, `capabilities`, `submit(job)`,
  `poll(ticket)`, `wait(ticket) -> int`, `quiesce()`. `hetero_kv.torch_engine.TorchCopyEngine`
  uses `Tensor.copy_` and CUDA events so Torch programs can use the lifecycle before the native
  CUDA engine (P1) exists.

Later phases add engines behind features: CUDA (P1, CUDA driver API, pinned staging, own
stream), RNGD PDMA <-> DMA-BUF (P2, RNGD kernel driver UAPI), NIXL (P3).

## Wire format

Descriptors (`RegionDesc`, `KvDescriptor`, `AgentMeta`) encode as
`{"hkv": 1, "type": "<kind>", "body": {...}}` JSON. Decoders reject other versions and kinds;
layouts are re-validated on decode.
