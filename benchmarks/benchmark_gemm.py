import torch
import triton

from utils import add_tritonblas_lib

add_tritonblas_lib()
import tritonblas as tb


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "n", "k"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="gemm-performance",
        args={},
    )
)
def benchmark_gemm(m, n, k, provider):
    a = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
    b = torch.randn((k, n), device=DEVICE, dtype=torch.float16)
    c = torch.randn((m, n), device=DEVICE, dtype=torch.float16)

    alpha = 0.5
    beta = 100
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.addmm(c, a, b, alpha=alpha, beta=beta), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tb.gemm(a, b, c=c, alpha=alpha, beta=beta), quantiles=quantiles
        )

    def perf(ms):
        return 2 * m * n * k * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    benchmark_gemm.run(print_data=True, show_plots=True)
