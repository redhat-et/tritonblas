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
def gemv_kernel(
        # pointers
        a_ptr,
        x_ptr,
        y_ptr,
        # scalars
        alpha,
        beta,
        output,
        # dimensions
        M,
        N,
        # strides
        stride_am,
        stride_an,
        stride_x,
        stride_y,
        # block size
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offset_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offset_n = tl.arange(0, BLOCK_SIZE_N)
    mask = offset_m < M and offset_n < N
    a = tl.load(a_ptr + offset_m * stride_am + offset_n * stride_an, mask=mask)
    x = tl.load(x_ptr + offset_n * stride_x, mask=mask)
    output = tl.dot(a, x)
    tl.store(y_ptr + offset_m * stride_y, output, mask=mask)


def gemv(a: torch.Tensor, x: torch.Tensor):
    assert a.device == x.device
    assert a.shape[1] == x.shape[0]
    m, n = a.shape
    output = torch.empty_like((n))
    n_elements = output.numel()

    def grid(META): return (triton.cdiv(n_elements, META['BLOCK_SIZE']), )

    gemv_kernel[grid](
        a, x, 
        output, 
        n_elements
    )