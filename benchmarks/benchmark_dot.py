import torch
import triton

from utils import add_tritonblas_lib

add_tritonblas_lib()
import tritonblas as tb


DEVICE = "cuda"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        # Different possible values for `x_name`.
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,  # x axis is logarithmic.
        # Argument name whose value corresponds to a different line in the plot.
        line_arg='provider',
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        # Name for the plot. Used also as a file name for saving the plot.
        plot_name='vector-add-performance',
        # Values for function arguments not in `x_names` and `y_name`.
        args={},
    ))
def benchmark_dot(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x * y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tb.dot(x, y), quantiles=quantiles)

    def gbps(ms): return 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def test_dot(benchmark):
    def run_dot_benchmark():
        benchmark_dot.run(print_data=False, show_plots=False)

    benchmark(run_dot_benchmark)


if __name__ == "__main__":
    benchmark_dot.run(print_data=True, show_plots=False)
