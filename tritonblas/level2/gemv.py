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
    block_start