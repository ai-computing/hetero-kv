# NIXL backend plugin (design notes, phase P3)

Goal: let NIXL users (vLLM `NixlConnector`, Dynamo KVBM, llm-d) move KV over hetero-kv engines
(RNGD PDMA, ATOM, host) by loading a NIXL backend plugin, without changing the consumer.

Shape: a thin C++ `nixlBackendEngine` implementation over the C ABI (`include/hetero_kv.h`); no
Rust in the plugin itself.

| NIXL backend call | hetero-kv |
|---|---|
| `getSupportedMems()` | memory kinds of the engine's `Capabilities` (`DRAM_SEG` = host, `VRAM_SEG` = cuda/rngd/atom) |
| `registerMem(desc, mem, out)` | wrap the range as a region at `hkv_location{kind, devId}` (`hkv_region_wrap`) and register it (`hkv_agent_export`) |
| `getPublicData(meta)` / `loadRemoteMD` | region / agent descriptors (`RegionDesc`, `AgentMeta` wire bytes) |
| `prepXfer(op, local, remote, agent)` | a span list; NIXL descriptors are byte ranges, so no KV layout is involved |
| `postXfer(...)` | `Route::submit` (L2 transfer handle); `NIXL_READ` / `NIXL_WRITE` = `Direction::Read` / `Write` |
| `checkXfer(handle)` | `Transfer::poll` (`NIXL_IN_PROG` / `NIXL_SUCCESS`; an error is uncertain) |
| `releaseReqH` | drop the transfer; dropping an unfinished one marks its regions uncertain |
| notifications (`genNotif`, `getNotifs`) | P3 cross-process notification channel |

Already in the C ABI: locations, agent registration and descriptor export/import as bytes,
Selection-based session transfers. Still needed (not in P0): raw span submission at L2,
transfer polling without a session, C-implemented engines, and cross-process resolution.

Rule: the plugin talks to devices only through hetero-kv engines, which use documented
interfaces only (see ARCHITECTURE.md).
