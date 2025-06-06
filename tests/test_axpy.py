import torch
import tritonblas as tb
import triton
from triton.testing import assert_close
from .utils import is_hip_mi200, get_device


def test_axpy():
    size = 98432
    DEVICE = get_device()

    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.randn(size, device=DEVICE, dtype=dtype)
    y = torch.randn(size, device=DEVICE, dtype=dtype)
    alpha = 0.42

    triton_output = tb.axpy(x, y, alpha)
    torch_output = (alpha * x) + y

    rtol = 1e-2 if is_hip_mi200() else 0
    assert_close(triton_output, torch_output, rtol=rtol)
