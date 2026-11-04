# Roadmap

Where hetero-kv is going, phase by phase. Design and rules: [ARCHITECTURE.md](ARCHITECTURE.md),
Issue [#1](https://github.com/ai-computing/hetero-kv/issues/1). How consumers map onto the API:
[INTEGRATION.md](INTEGRATION.md). Checked items are on `main`.

## P0 — Core (host only)

- [x] Cargo workspace: core, C ABI (`libhetero_kv`, generated header), Python (`hetero_kv`) (#3)
- [x] L1 regions with byte-range ownership; Uncertain taints the region until quiesce (#3)
- [x] Device locations (`cuda:1`, `rngd:3`, `atom:2@node`) in regions, descriptors and routes (#3)
- [x] L2 `CopyEngine`, capability matrix, declared representation conversions, no fallback (#3)
- [x] L3 `KvSpec`/`Layout` presets, block-id and prefix selections, layer progress, block hashes, KV events (#3)
- [x] L4 per-request `Session`: reserve → transfer → fence → admit → finish, cancel, poison, recover (#3)
- [x] Host memcpy engine; Python-implemented engines; Torch copy engine (#3)
- [x] Optimus Prime uses hetero-kv; glue lives in aicomp `accel/pd/transport.py` (aicomp#299, #6)
- [ ] Concurrent reads of one source range (reader/writer claims) (#4)
- [ ] Prebuilt manylinux wheels from CI so consumers don't need a Rust toolchain
- [ ] Versioning: semver tags, CHANGELOG, wire-format compatibility policy
- [ ] License decision
- [ ] Property tests for range ownership and plan coverage; `cargo miri` on L1

## P1 — CUDA

- [ ] CUDA engine on the CUDA driver API: own stream, events as completions, `quiesce` = stream sync
- [ ] Pinned host staging pool with bounded, request-owned slots
- [ ] Routes cuda ↔ host / host_pinned, cuda ↔ cuda (same device; peer only where P2P is verified correct)
- [ ] Per-layer import overlapping decode on a separate stream; fenced on admission, error, cancel, slot reuse (aicomp#269)
- [ ] CUDA memory as a DMA-BUF region (export for other devices' DMA engines)
- [ ] GPU CI job (self-hosted runner) running the CUDA tests
- [ ] Optimus `TorchCacheTransport` moves from the Torch copy engine to the CUDA engine; bit-identical A/B

## P2 — NPU engines

- [ ] RNGD PDMA engine on the documented driver UAPI (io_uring commands), ported from hetero-npu-p2p `rngd_dmabuf.c`
- [ ] RNGD ↔ CUDA through CUDA DMA-BUF (measured 19 GB/s RNGD→GPU, 10–13 GB/s GPU→RNGD) (aicomp#121)
- [ ] ATOM DMA-BUF destination regions; RNGD → ATOM route
- [ ] Declared BF16 → DLFloat16 conversion route for RNGD → ATOM
- [ ] Hardware validation on the RNGD+ATOM host: byte-exact transfers, Optimus `validate_koni.py` token/logit parity
- [ ] Optimus `p2p_transport` uses the native engine instead of the hetero_p2p backend; hetero-npu-p2p archived
- [ ] Transfer-curve benchmarks per route (size × queue depth), checked into `docs/`

## P3 — Across processes and hosts

- [ ] Descriptor resolution across processes: DMA-BUF / memfd fd passing over a Unix socket
- [ ] Remote completion notification (producer learns when the consumer has fenced, and vice versa)
- [ ] Optimus `npu/shared_kv.SharedKVSlab` replaced by a hetero-kv host region shared by fd
- [ ] NIXL backend plugin: thin C++ shim over the C ABI
- [ ] NIXL as an engine (UCX / RDMA) for cross-host routes (`@node` locations)
- [ ] C ABI: span-level submission, KV events, engines written in C

## P4 — Ecosystem

- [ ] vLLM V1 `KVConnector` (push and pull), tested against a pinned vLLM release
- [ ] llm-d: `kv_transfer_params` carrying hetero-kv descriptors through the P/D sidecar; KVEvents publisher example
- [ ] LMCache storage backend and Mooncake transfer-engine backend
- [ ] Dynamo KVBM layouts and tiers (G1–G4) end to end
- [ ] Storage tier engine (file / NVMe)
- [ ] rngd-pd and ai-rm move their KV transports onto hetero-kv; ai-rm engine-integration doc updated

## P5 — Operations

- [ ] Metrics: bytes, latency and in-flight transfers per route; Prometheus names aligned with `airm_*`
- [ ] Tracing hooks (submit / complete / fence / admit spans)
- [ ] Fault injection for engines (short completion, lost completion, quiesce failure)
- [ ] Documentation site generated from the crate docs and these guides
