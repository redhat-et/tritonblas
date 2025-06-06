import triton

def get_device():
    return triton.runtime.driver.active.get_current_target().backend

# shared rtol function
def is_hip_mi200():
    DEVICE = get_device()
    arch = triton.runtime.driver.active.get_current_target().arch
    return DEVICE == "hip" and arch == "gfx90a"
