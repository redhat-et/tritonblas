import torch
import triton
import tritonblas as tb
from triton.testing import assert_close
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_gemv():
    m, n = 512, 64

    torch.manual_seed(0)
    dtype = torch.float32
    a = torch.randn(m, n, device=DEVICE, dtype=dtype)
    x = torch.randn(n, device=DEVICE, dtype=dtype)
    y = torch.randn(m, device=DEVICE, dtype=dtype)
    alpha = 0.42
    beta = 10

    triton_output = tb.gemv(a, x, y, alpha, beta)
    torch_output = alpha * torch.mv(a, x) + beta * y

    rtol = get_rtol()
    assert_close(triton_output, torch_output, rtol=rtol)

