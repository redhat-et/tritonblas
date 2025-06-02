import torch
import triton
from triton.testing import assert_close
import tritonblas as tb


DEVICE = "cuda"

def test_nrm2():
    size = 98432

    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE, dtype=torch.float16)

    triton_output = tb.nrm2(x)
    torch.output = torch.linalg.norm(x)
    
    assert_close(triton_output, torch_output, rtol=rtol)
