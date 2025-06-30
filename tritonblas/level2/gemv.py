import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_K": 64,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_K": 64,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_K": 64,
            },
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "K"],
)
@triton.jit
def gemv_kernel(
    # Pointers to matrices
    a_ptr,
    x_ptr,
    # Output pointer
    y_ptr,
    # Matrix dimensions
    M,
    K,
    # Stride lengths
    stride_am,
    stride_ak,
    stride_x,
    stride_y,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the GEMV, y = A * x.
    A has shape (M, K), x has shape (K,) and y has shape (M,).
    """
    pid = tl.program_id(axis=0)
    
    offs_am = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_block = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    x_block = x_ptr + offs_k * stride_x

    # Get the product of A*x
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Boundary check for K dimension
        k_mask = offs_k < K - k * BLOCK_SIZE_K
        a_mask = (offs_am[:, None] < M) & k_mask[None, :]
        
        tiled_a = tl.load(a_block, mask=a_mask, other=0.0)
        tiled_x = tl.load(x_block, mask=k_mask, other=0.0)
        
        # Compute partial dot product for each row
        partial_result = tl.sum(tiled_a * tiled_x[None, :], axis=1)
        accumulator += partial_result
        
        a_block += BLOCK_SIZE_K * stride_ak
        x_block += BLOCK_SIZE_K * stride_x

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_block = y_ptr + offs_m * stride_y
    y_mask = offs_m < M

    tl.store(y_block, accumulator, mask=y_mask)


def gemv(a: torch.Tensor, x: torch.Tensor):
    assert a.shape[1] == x.shape[0]
    M, K = a.shape

    y = torch.empty((M,), device=a.device, dtype=a.dtype)

    def grid(META): return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    gemv_kernel[grid](
        a,
        x,
        y,
        M,
        K,
        a.stride(0),
        a.stride(1),
        x.stride(0),
        y.stride(0),
    )

    return y