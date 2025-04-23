import torch
import tritonblas as tb
import triton
from triton.testing import assert_close


DEVICE = "cuda"

# TODO either make this shared or make a shared rtol function
def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

def test_axpy():
    size = 98432

    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.randn(size, device=DEVICE, dtype=dtype)
    y = torch.randn(size, device=DEVICE, dtype=dtype)
    alpha = 0.42

    triton_output = tb.axpy(x, y, alpha)
    torch_output = (alpha * x) + y
    
    rtol = 1e-2 if is_hip_mi200() else 0
    assert_close(triton_output, torch_output, rtol=rtol)
