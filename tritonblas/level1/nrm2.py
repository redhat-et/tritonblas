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


def next_power_of_2(n):
    """Return the next power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


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
    """Kernel to compute the partial sums of the squares of the elements of the input tensor.

    Args:
        x_ptr: Pointer to the input tensor.
        partial_ptr: Pointer to the partial sums.
        n_elements: Number of elements in the input tensor.
        BLOCK_SIZE: Block size autotuned by the compiler.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    exp = x * x
    sum_of_squares = tl.sum(exp)
    
    tl.store(partial_ptr + pid, sum_of_squares)


@triton.jit
def nrm2_final(
    partial_ptr,
    final_ptr,
    n_partial,
    BLOCK_SIZE_FINAL: tl.constexpr,
):
    """Kernel to compute the final sum of the partial sums using parallel reduction.

    Args:
        partial_ptr: Pointer to the partial sums.
        final_ptr: Pointer to the final sum.
        n_partial: Number of partial sums calculated by the partial kernel.
        BLOCK_SIZE_FINAL: Block size for loading partial sums (should equal n_partial).
    """
    offsets = tl.arange(0, BLOCK_SIZE_FINAL)
    mask = offsets < n_partial
    partial_vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    
    total = tl.sum(partial_vals)
    output = tl.sqrt(total)
    tl.store(final_ptr, output)


def nrm2(x: torch.Tensor):
    """Compute the L2 norm of a tensor using a two-kernel parallel reduction.

    Workflow:
    1. Partial Kernel (Distributed): Divides the input vector into blocks,
       where each GPU thread block computes the sum of squares for its assigned
       chunk of elements. This produces multiple partial sums.
       
    2. Final Kernel (Reduction): A single thread sequentially sums all the
       partial sums from step 1 to get the total sum of squares.
       
    3. Square Root: Takes the square root of the total sum to get the L2 norm.

    Args:
        x: Input tensor.

    Returns:
        L2 norm of the input tensor (scalar).
    """
    output = torch.zeros((1,), device=x.device)
    assert x.device == output.device
    n_elements = x.numel()

    # Allocate enough memory for the partial sums
    min_block_size = min(config.kwargs['BLOCK_SIZE'] for config in get_autotune_config())
    max_partial_programs = triton.cdiv(n_elements, min_block_size)
    partial_sums = torch.zeros(max_partial_programs, device=x.device)

    # Capture the actual grid size used by autotuning
    auto_grid = None
    def grid(META):
        nonlocal auto_grid
        auto_grid = triton.cdiv(n_elements, META['BLOCK_SIZE'])
        return (auto_grid,)
    
    nrm2_partial[grid](
        x,
        partial_sums,
        n_elements
    )

    # Round up auto_grid to next power of 2 for Triton compatibility
    block_size_final = next_power_of_2(auto_grid)
    
    nrm2_final[(1,)](
        partial_sums,
        output,
        auto_grid,
        BLOCK_SIZE_FINAL=block_size_final # needs to be a power of 2 for Triton compatibility
    )

    return output
