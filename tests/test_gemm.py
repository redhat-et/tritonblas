import pytest
import torch
import triton
import tritonblas as tb
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@pytest.mark.parametrize(
    "m, n, k",
    [
        (512, 512, 512),
    ]
)
@pytest.mark.parametrize(
    "alpha, beta",
    [
        (0, 0),
        (0.5, 100),
    ]
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
    ]
)
def test_gemm(m, n, k, alpha, beta, dtype):
    torch.manual_seed(0)
    a = torch.randn((m, k), device=DEVICE, dtype=dtype)
    b = torch.randn((k, n), device=DEVICE, dtype=dtype)
    c = torch.randn((m, n), device=DEVICE, dtype=dtype)

    triton_output = tb.gemm(a, b, c=c, alpha=alpha, beta=beta)
    torch_output = torch.addmm(c, a, b, alpha=alpha, beta=beta)
    triton.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=get_rtol())
