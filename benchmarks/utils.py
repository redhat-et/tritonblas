# Import the tritonblas library without installing it
def add_tritonblas_lib():
    import os
    import sys

    dir_path = os.path.dirname(os.path.realpath(__file__ + "/../"))
    sys.path.append(dir_path)
