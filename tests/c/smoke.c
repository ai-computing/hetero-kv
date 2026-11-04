/* C smoke test: reserve -> transfer -> fence -> admit -> finish with the host engine; route
 * rejection; agent registration, descriptor export/import as bytes and a pull through the
 * resolved descriptor; layer progress; reclaim. Build: see tests/c/run.sh. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hetero_kv.h"

#define CHECK(expr)                                                                   \
  do {                                                                                \
    hkv_status s_ = (expr);                                                           \
    if (s_ != HKV_OK) {                                                               \
      fprintf(stderr, "%s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #expr, (int)s_,   \
              hkv_last_error());                                                      \
      return 1;                                                                       \
    }                                                                                 \
  } while (0)

static int notified = 0;
static hkv_status on_admit(void *user) {
  *(int *)user += 1;
  return HKV_OK;
}

static hkv_buffer *make_buffer(const hkv_spec *spec, unsigned char seed) {
  size_t n = hkv_spec_region_count(spec);
  uint64_t bytes = hkv_spec_region_bytes(spec);
  hkv_region *regions[16];
  for (size_t i = 0; i < n; i++) {
    regions[i] = hkv_region_host_new((size_t)bytes);
    uint8_t *p = hkv_region_data(regions[i]);
    for (uint64_t j = 0; j < bytes; j++) p[j] = (uint8_t)(seed + i * 31 + j);
  }
  return hkv_buffer_new(spec, regions, n);
}

int main(void) {
  printf("hetero-kv %s\n", hkv_version());
  hkv_geometry g = {HKV_BF16, 0, 2, 1, 8, 2, 4}; /* 2 layers, 1 block of 8 tokens, 2 heads, dim 4 */
  hkv_spec *spec = hkv_spec_new("model", "rev", HKV_LAYOUT_BHSD, &g);
  if (!spec) return fprintf(stderr, "spec: %s\n", hkv_last_error()), 1;

  hkv_engine *engine = hkv_host_engine_new();
  hkv_location host = {HKV_MEM_HOST, 0, false, 0}, cuda1 = {HKV_MEM_CUDA, 1, false, 0};
  /* No silent fallback: the host engine has no host -> cuda:1 route, and no conversions. */
  if (hkv_session_new(engine, host, cuda1, HKV_BF16, HKV_BF16) != NULL) return 1;
  printf("rejected as expected: %s\n", hkv_last_error());
  if (hkv_session_new(engine, host, host, HKV_BF16, HKV_DLFLOAT16) != NULL) return 1;

  hkv_session *session = hkv_session_new(engine, host, host, HKV_BF16, HKV_BF16);
  hkv_buffer *src = make_buffer(spec, 7), *dst = make_buffer(spec, 0);
  if (!session || !src || !dst) return 1;

  const hkv_buffer *dsts[1] = {dst};
  CHECK(hkv_session_reserve(session, "req-1", dsts, 1));
  if (hkv_session_admit(session, "req-1", on_admit, &notified) != HKV_STATE) return 1; /* not fenced */
  uint32_t block0 = 0;
  hkv_selection sel = {0, 2, &block0, 1, &block0, 1, 0, 0, 5}; /* first 5 tokens, all layers */
  uint64_t id = 0;
  CHECK(hkv_session_transfer(session, "req-1", 0, src, &sel, 1, 0, &id));
  hkv_receipt receipt;
  CHECK(hkv_session_fence(session, "req-1", &receipt));
  CHECK(hkv_session_admit(session, "req-1", on_admit, &notified));
  CHECK(hkv_session_finish(session, "req-1", true));
  if (notified != 1 || receipt.bytes != 2 * 2 * 2 * 5 * 8 || hkv_session_get_state(session) != HKV_SESSION_IDLE) {
    fprintf(stderr, "unexpected receipt %llu / notified %d\n", (unsigned long long)receipt.bytes, notified);
    return 1;
  }

  /* Pull through a descriptor: export src, ship it as bytes, resolve it on the "decode" side and
   * copy the remaining tokens 5..8 with direction = read. */
  hkv_agent *prefill = hkv_agent_new("prefill", "prefill:5555");
  uint32_t hash_block = 0;
  uint64_t hash = 42;
  hkv_descriptor *desc = hkv_agent_export(prefill, src, &hash_block, &hash, 1);
  if (!desc) return fprintf(stderr, "export: %s\n", hkv_last_error()), 1;
  size_t len = 0;
  if (hkv_descriptor_to_bytes(desc, NULL, 0, &len) != HKV_INVALID || len == 0) return 1;
  uint8_t *wire = malloc(len);
  CHECK(hkv_descriptor_to_bytes(desc, wire, len, &len));
  size_t meta_len = 0;
  hkv_agent_meta(prefill, NULL, 0, &meta_len);
  if (meta_len == 0) return 1;

  hkv_descriptor *received = hkv_descriptor_from_bytes(wire, len);
  hkv_buffer *remote_src = received ? hkv_descriptor_resolve(received) : NULL;
  if (!remote_src) return fprintf(stderr, "resolve: %s\n", hkv_last_error()), 1;
  hkv_region *none[1];
  if (hkv_buffer_reclaim(remote_src, none, 1) != HKV_STATE) return 1; /* references have no regions */
  CHECK(hkv_session_reserve(session, "req-2", dsts, 1));
  hkv_selection rest = {0, 2, &block0, 1, &block0, 1, 5, 5, 3};
  CHECK(hkv_session_transfer(session, "req-2", 0, remote_src, &rest, 1, 1, &id));
  CHECK(hkv_session_fence(session, "req-2", &receipt));
  CHECK(hkv_session_admit(session, "req-2", NULL, NULL));
  CHECK(hkv_session_finish(session, "req-2", true));
  if (receipt.bytes != 2 * 2 * 2 * 3 * 8) return 1;
  hkv_buffer_free(remote_src);
  hkv_descriptor_free(received);
  hkv_region *a[4], *b[4];
  if (hkv_buffer_reclaim(src, a, 4) != HKV_BUSY) return 1; /* still registered */
  CHECK(hkv_agent_release(prefill, desc));
  hkv_descriptor_free(desc);
  hkv_agent_free(prefill);
  free(wire);

  /* Verify: each head's 8 rows (8 * 8 bytes) of each region match. */
  CHECK(hkv_buffer_reclaim(src, a, 4));
  CHECK(hkv_buffer_reclaim(dst, b, 4));
  for (int r = 0; r < 4; r++)
    for (int h = 0; h < 2; h++)
      if (memcmp(hkv_region_data(a[r]) + h * 64, hkv_region_data(b[r]) + h * 64, 64) != 0) return 1;
  for (int r = 0; r < 4; r++) {
    hkv_region_free(a[r]);
    hkv_region_free(b[r]);
  }

  /* Layer progress: overlap rejected, coverage enforced. */
  hkv_progress *p = hkv_progress_new(2, 0, 4);
  CHECK(hkv_progress_publish(p, 0, 2, 0, 2));
  if (hkv_progress_publish(p, 1, 2, 1, 3) != HKV_INVALID) return 1;
  if (hkv_progress_finish(p) != HKV_INVALID) return 1;
  CHECK(hkv_progress_publish(p, 0, 2, 2, 4));
  hkv_progress_span spans[1];
  size_t n = 0;
  CHECK(hkv_progress_drain(p, spans, 1, &n));
  if (n != 1 || !hkv_progress_pending(p)) return 1;
  CHECK(hkv_progress_drain(p, spans, 1, &n));
  if (n != 1 || spans[0].token_start != 2 || hkv_progress_pending(p)) return 1;
  CHECK(hkv_progress_finish(p));
  hkv_progress_free(p);

  hkv_session_free(session);
  hkv_engine_free(engine);
  hkv_spec_free(spec);
  printf("C smoke test passed\n");
  return 0;
}
