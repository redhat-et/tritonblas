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
def nrm2_partial(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Kernel to compute the partial sums of the squares of the elements of the input tensor."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    exp = x * x
    sum_of_squares = tl.sum(exp)
    
    tl.store(partial_ptr + pid, sum_of_squares)


@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_partial'])})
@triton.jit
def nrm2_final(
    partial_ptr,
    final_ptr,
    n_partial,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to compute the final sum of the partial sums using parallel reduction."""

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial
    partial_vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    
    total = tl.sum(partial_vals)
    output = tl.sqrt(total)
    tl.store(final_ptr, output)


def nrm2(x: torch.Tensor):
    """Compute the L2 norm of a tensor using a two-kernel parallel reduction."""
    output = torch.zeros((1,), device=x.device)
    assert x.device == output.device
    n_elements = x.numel()

    # Allocate memory for partial sums
    min_block_size = min(config.kwargs['BLOCK_SIZE'] for config in get_autotune_config())
    max_partial_programs = triton.cdiv(n_elements, min_block_size)
    partial_sums = torch.zeros(max_partial_programs, device=x.device)

    # Get number of partial sums calculated by the partial kernel
    # to use in the heuristics for the final kernel
    n_partials = None
    def grid(META):
        nonlocal n_partials
        n_partials = triton.cdiv(n_elements, META['BLOCK_SIZE'])
        return (n_partials,)

    nrm2_partial[grid](
        x,
        partial_sums,
        n_elements
    )

    nrm2_final[(1,)](
        partial_sums,
        output,
        n_partials
    )

    return output
