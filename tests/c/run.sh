#!/bin/sh
# Build the C ABI, compile the smoke test against the header and cdylib, run it.
set -eu
cd "$(dirname "$0")/../.."
cargo build -p hetero-kv-capi
out=target/debug
${CC:-cc} -std=c11 -Wall -Wextra -Werror -Iinclude tests/c/smoke.c -L"$out" -lhetero_kv -Wl,-rpath,"$PWD/$out" -o "$out/hkv_smoke"
"$out/hkv_smoke"
