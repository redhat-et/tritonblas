import torch
import triton
from triton.testing import assert_close
from .utils import get_rtol

# Import the gemv function directly since it's not exported in the main module yet
from tritonblas.level2.gemv import gemv


DEVICE = triton.runtime.driver.active.get_current_target().backend


def test_gemv():
    m, n = 512, 64

    torch.manual_seed(0)
    dtype = torch.float32
    a = torch.randn(m, n, device=DEVICE, dtype=dtype)
    x = torch.randn(n, device=DEVICE, dtype=dtype)

    triton_output = gemv(a, x)
    torch_output = torch.matmul(a, x)

    rtol = get_rtol()
    assert_close(triton_output, torch_output, rtol=rtol)

