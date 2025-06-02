import torch
import triton
import tritonblas as tb
from triton.testing import assert_close


DEVICE = triton.runtime.driver.active.get_current_target().backend


def is_hip_mi200():
    return DEVICE == "hip" and triton.runtime.driver.active.get_current_target().arch == "gfx90a"


def test_dot():
    size = 98432

    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE, dtype=torch.float16)
    y = torch.randn(size, device=DEVICE, dtype=torch.float16)

    triton_output = tb.dot(x, y)
    torch_output = x * y

    rtol = 1e-2 if is_hip_mi200() else 0
    assert_close(triton_output, torch_output, rtol=rtol)
