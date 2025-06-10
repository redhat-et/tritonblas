import torch
import triton
from triton.testing import assert_close
import tritonblas as tb
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_current_target().backend


def test_nrm2():
    size = 98432

    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.randn(size, device=DEVICE, dtype=dtype)

    triton_output = tb.nrm2(x)
    torch_output = torch.linalg.norm(x)
    
    rtol = get_rtol()
    assert_close(triton_output, torch_output, rtol=rtol)
