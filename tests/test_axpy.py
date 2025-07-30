import torch
import tritonblas as tb
import triton
from triton.testing import assert_close
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_axpy():
    size = 98432

    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.randn(size, device=DEVICE, dtype=dtype)
    y = torch.randn(size, device=DEVICE, dtype=dtype)
    alpha = 0.42

    triton_output = tb.axpy(x, y, alpha)
    torch_output = (alpha * x) + y

    rtol = get_rtol()
    assert_close(triton_output, torch_output, rtol=rtol)
