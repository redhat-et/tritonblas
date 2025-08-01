import torch
import triton
import tritonblas as tb
from triton.testing import assert_close
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_nrm2():
    size = 98432

    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.randn(size, device=DEVICE, dtype=dtype)

    triton_output = tb.nrm2(x)
    torch_output = torch.linalg.norm(x)
    
    rtol = get_rtol()
    assert_close(triton_output, torch_output, rtol=rtol)
