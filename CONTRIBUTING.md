# Contributing

Follow the organization guide
([ai-computing/.github CONTRIBUTING.md](https://github.com/ai-computing/.github/blob/main/CONTRIBUTING.md)):
issue first, one branch per change (`<type>/<issue#>-<slug>`), one purpose per PR, the maintainer
merges.

## Rules specific to hetero-kv

1. **Documented interfaces only.** Engines and adapters may use kernel driver UAPI, public vendor
   SDK APIs, DMA-BUF and the CUDA driver API. Do not hook vendor binaries (LD_PRELOAD shims,
   symbol interposition, binary patching, private ioctls reverse-engineered from closed
   libraries), and do not depend on patched or hash-pinned third-party binaries. A PR that needs
   one of these is out of scope for this repository.
2. **Keep the layering.** L4 session -> L3 kv -> L2 engine -> L1 memory; never upwards. Vendor
   code lives behind its cargo feature (`cuda`, `rngd`, `atom`, `nixl`) inside an engine.
3. **No silent fallback.** A missing route is an error, never a host copy.
4. **Poison on uncertainty.** If a change cannot prove a copy finished, the regions must end up
   `Uncertain` and the session poisoned.
5. **CPU CI stays green.** New behaviour gets a test that runs without hardware (host engine,
   Python engine or a scripted fake). Hardware results go in the PR with commit SHAs, hardware and
   SDK versions, and how to reproduce.
6. **C ABI changes** regenerate `include/hetero_kv.h` (`scripts/gen-header.sh`) in the same PR.
7. Name hosts by role, not by hostname or IP, and keep personal paths out of committed files.

## Checks

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude hetero-kv-py
(cd crates/hetero-kv-py && maturin develop) && pytest tests/python adapters/optimus/tests
tests/c/run.sh
```
