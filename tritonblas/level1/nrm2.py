import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE": 1024,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 524,
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
def nrm2_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    partial = tl.sum(x**2)
    total = tl.reduce(partial, axis=0, op=tl.sum)
    output = tl.sqrt(total)

    tl.store(output_ptr, output)


def nrm2(x: torch.Tensor, y: torch.Tensor):
    output = torch.Tensor()
    assert x.device == y.device and x.device == output.device
    n_elements = output.numel()

    def grid(META): return (triton.cdiv(n_elements, META['BLOCK_SIZE']), )

    nrm2_kernel[grid](
        x,
        output,
        n_elements
    )
    
    return output