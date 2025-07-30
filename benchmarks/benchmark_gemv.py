import torch
import triton

from utils import add_tritonblas_lib

add_tritonblas_lib()
import tritonblas as tb


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # Argument names to use as an x-axis for the plot
        x_names=["M", "K"],
        # Different possible values for `x_name`
        x_vals=[128 * i for i in range(2, 33)],
        # Argument name whose value corresponds to a different line in the plot
        line_arg="provider",
        # Possible values for `line_arg`
        line_vals=["triton", "torch"],
        # Label name for the lines
        line_names=["Triton", "PyTorch"],
        # Line styles
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",  # Label name for the y-axis
        plot_name="gemv-performance",
        args={},
    )
)
def benchmark_gemv(M, K, provider):
    A = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    x = torch.randn(K, device=DEVICE, dtype=torch.float16)
    y = torch.randn(M, device=DEVICE, dtype=torch.float16)
    alpha = 0.42
    beta = 10
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: alpha * torch.mv(A, x) + beta * y, quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tb.gemv(A, x, y, alpha, beta), quantiles=quantiles
        )

    # Calculate GB/s: reads A (M*K) + x (K) + y (M) + writes output (M)
    total_elements = M * K + K + M + M
    def gbps(ms): return total_elements * A.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def test_gemv(benchmark):
    def run_gemv_benchmark():
        benchmark_gemv.run(print_data=False, show_plots=False)

    benchmark(run_gemv_benchmark)


if __name__ == "__main__":
    benchmark_gemv.run(print_data=True, show_plots=False) 