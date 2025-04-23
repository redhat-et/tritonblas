import torch
import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE": 1024,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 512,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 256,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 128,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 64,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 32,
            },
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["n_elements"]
)
@triton.jit
def axpy_kernel(
        x_ptr, y_ptr,
        alpha,
        output_ptr, # TODO output should overwrite y
        n_elements,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = (alpha * x) + y

    tl.store(output_ptr + offsets, output, mask=mask)


def axpy(x: torch.Tensor, y: torch.Tensor, alpha):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    def grid(META): return (triton.cdiv(n_elements, META['BLOCK_SIZE']), )

    axpy_kernel[grid](
        x, y,
        alpha,
        output,
        n_elements
    )

    return output
