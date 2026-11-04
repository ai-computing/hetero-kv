#!/bin/sh
# Regenerate include/hetero_kv.h from the C ABI crate (needs cbindgen on PATH).
set -eu
cd "$(dirname "$0")/.."
cbindgen --config crates/hetero-kv-capi/cbindgen.toml --crate hetero-kv-capi --output include/hetero_kv.h
