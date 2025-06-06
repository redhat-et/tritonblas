import torch
import triton
import tritonblas as tb
from triton.testing import assert_close
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_current_target().backend


def test_dot():
    size = 98432

    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE, dtype=torch.float16)
    y = torch.randn(size, device=DEVICE, dtype=torch.float16)

    triton_output = tb.dot(x, y)
    torch_output = x * y

    rtol = get_rtol()
    assert_close(triton_output, torch_output, rtol=rtol)
