import torch
import tritonblas as tb


DEVICE = "cuda"


def test_dot():
    size = 98432

    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE, dtype=torch.float16)
    y = torch.randn(size, device=DEVICE, dtype=torch.float16)

    triton_output = tb.dot(x, y)
    torch_output = x * y
    
    assert torch.max(torch.abs(torch_output - triton_output)) == 0
