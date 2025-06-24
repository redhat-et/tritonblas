import torch
import triton
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
        plot_name='vector-nrm2-performance',
        # Values for function arguments not in `x_names` and `y_name`.
        args={},
    ))
def benchmark_nrm2(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.linalg.norm(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tb.nrm2(x), quantiles=quantiles)

    def gbps(ms): return 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def test_nrm2(benchmark):
    def run_nrm2_benchmark():
        benchmark_nrm2.run(print_data=False, show_plots=False)

    benchmark(run_nrm2_benchmark)


if __name__ == "__main__":
    benchmark_nrm2.run(print_data=True, show_plots=False) 