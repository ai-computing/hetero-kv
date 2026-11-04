"""TorchCopyEngine: a Torch `copy_` engine plugged into the Rust session (skips without torch)."""
import pytest

torch = pytest.importorskip("torch")

import hetero_kv as hk  # noqa: E402
from hetero_kv.torch_engine import TorchCopyEngine, tensor_region  # noqa: E402


def tensors_buffer(spec, tensors):
    return hk.KVBuffer(spec, [tensor_region(t) for t in tensors])


def bhsd_spec(layers, heads, cap, dim, dtype="bfloat16"):
    return hk.KVSpec("tiny", "seed", hk.Layout.preset("bhsd", dtype, layers, cap, heads, dim))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_torch_engine_prefix_copy_matches_tensor_copy(dtype):
    torch.manual_seed(0)
    layers, heads, dim, n = 3, 2, 8, 11
    src = [torch.randn(1, heads, 16, dim, dtype=dtype) for _ in range(2 * layers)]
    dst = [torch.zeros(1, heads, 32, dim, dtype=dtype) for _ in range(2 * layers)]
    rep = str(dtype)
    s_spec, d_spec = bhsd_spec(layers, heads, 16, dim, rep), bhsd_spec(layers, heads, 32, dim, rep)
    session = hk.Session(TorchCopyEngine(), "host", "host", rep)
    receipt = hk.transfer_and_admit(session, "r", tensors_buffer(s_spec, src), tensors_buffer(d_spec, dst),
                                    hk.Selection.prefix(n))
    session.finish("r")
    assert receipt["bytes"] == sum(t[:, :, :n].numel() * t.element_size() for t in src)
    for a, b in zip(src, dst):
        torch.testing.assert_close(b[:, :, :n], a[:, :, :n], rtol=0, atol=0)
        assert not b[:, :, n:].any()


def test_tensor_regions_need_contiguous_tensors():
    with pytest.raises(ValueError):
        tensor_region(torch.zeros(4, 4).t())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_torch_engine_host_to_cuda():
    spec = bhsd_spec(1, 1, 8, 4, "float32")
    src = [torch.arange(32, dtype=torch.float32).reshape(1, 1, 8, 4) for _ in range(2)]
    dst = [torch.zeros(1, 1, 8, 4, device="cuda") for _ in range(2)]
    session = hk.Session(TorchCopyEngine(), "host", "cuda", "float32")
    hk.transfer_and_admit(session, "r", tensors_buffer(spec, src), tensors_buffer(spec, dst), hk.Selection.prefix(8))
    session.finish("r")
    assert torch.equal(dst[0].cpu(), src[0])
