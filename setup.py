from setuptools import find_packages, setup

setup(
    name="tritonblas",
    version="0.1.0",
    author="Craig Magina",
    author_email="cmagina@redhat.com",
    description="BLAS Trtion routines",
    packages=find_packages(),
    install_requires=["triton"],
)
