import triton

# shared rtol function
def get_rtol():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend == "hip" and target.arch == "gfx90a":
        return 1e-2
    else:
        return 0
