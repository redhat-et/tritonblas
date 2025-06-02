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
    
)