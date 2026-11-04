# NIXL backend plugin (design notes, phase P3)

Goal: let NIXL users (vLLM `NixlConnector`, Dynamo KVBM, llm-d) move KV over hetero-kv engines
(RNGD PDMA, ATOM, host) by loading a NIXL backend plugin, without changing the consumer.

Shape: a thin C++ `nixlBackendEngine` implementation over the C ABI (`include/hetero_kv.h`); no
Rust in the plugin itself.

| NIXL backend call | hetero-kv |
|---|---|
| `getSupportedMems()` | memory kinds of the engine's `Capabilities` (`DRAM_SEG` = host, `VRAM_SEG` = cuda/rngd/atom) |
| `registerMem(desc, mem, out)` | wrap the range as a region (`hkv_region_wrap`) and register it (agent registry) |
| `getPublicData(meta)` / `loadRemoteMD` | region / agent descriptors (`RegionDesc`, `AgentMeta` wire bytes) |
| `prepXfer(op, local, remote, agent)` | a span list; NIXL descriptors are byte ranges, so no KV layout is involved |
| `postXfer(...)` | `Route::submit` (L2 transfer handle); `NIXL_READ` / `NIXL_WRITE` = `Direction::Read` / `Write` |
| `checkXfer(handle)` | `Transfer::poll` (`NIXL_IN_PROG` / `NIXL_SUCCESS`; an error is uncertain) |
| `releaseReqH` | drop the transfer; dropping an unfinished one marks its regions uncertain |
| notifications (`genNotif`, `getNotifs`) | P3 cross-process notification channel |

Needed in the C ABI before this can be written (not in P0): region registration and descriptor
export/import, raw span submission at L2 (today the C ABI exposes the L4 session only),
transfer polling without a session, and cross-process resolution.

Rule: the plugin talks to devices only through hetero-kv engines, which use documented
interfaces only (see ARCHITECTURE.md).
