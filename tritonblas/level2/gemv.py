import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_K": 128,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_K": 128,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_K": 128,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_K": 64,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_K": 32,
            },
        ),
        
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_K": 128,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_K": 256,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_K": 128,
            },
        ),
        
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_K": 256,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_K": 512,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_K": 256,
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
    y_ptr,
    # Output pointer
    output_ptr,
    # Scalars
    alpha,
    beta,
    # Matrix dimensions
    M,
    K,
    # Stride lengths
    stride_am,
    stride_ak,
    stride_x,
    stride_y,
    stride_output,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offs_am = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_block = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    x_block = x_ptr + offs_k * stride_x

    # Compute dot product of row of A and entire x vector in k blocks
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K - k * BLOCK_SIZE_K
        a_mask = (offs_am[:, None] < M) & k_mask[None, :]
        
        # Load BLOCK_SIZE_M x BLOCK_SIZE_K elements from A and BLOCK_SIZE_K elements from x
        tiled_a = tl.load(a_block, mask=a_mask, other=0.0)
        tiled_x = tl.load(x_block, mask=k_mask, other=0.0)
        
        # Partial dot product
        accumulator += tl.sum(tiled_a * tiled_x[None, :], axis=1)
        
        a_block += BLOCK_SIZE_K * stride_ak
        x_block += BLOCK_SIZE_K * stride_x

    # Load y vector
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_block = y_ptr + offs_m * stride_y
    output_block = output_ptr + offs_m * stride_output
    m_mask = offs_m < M
    
    y_vals = tl.load(y_block, mask=m_mask, other=0.0)
    
    # Compute final result: α⋅A⋅x + β⋅y
    result = alpha * accumulator + beta * y_vals

    tl.store(output_block, result, mask=m_mask)


def gemv(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor = None, alpha: float = 1.0, beta: float = 0.0):
    assert a.shape[1] == x.shape[0]
    M, K = a.shape

    # Handle y vector
    if y is None:
        y = torch.zeros((M,), device=a.device, dtype=a.dtype)
    else:
        assert y.shape[0] == M

    output = torch.empty((M,), device=a.device, dtype=a.dtype)

    def grid(META): return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    gemv_kernel[grid](
        a,
        x,
        y,
        output,
        alpha,
        beta,
        M,
        K,
        a.stride(0),
        a.stride(1),
        x.stride(0),
        y.stride(0),
        output.stride(0),
    )

    return output