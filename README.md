# tritonblas
BLAS routines written in Triton

## Routines

### Level 1

* dot

### Level 2

### Level 3

## Testing

The tritonblas project currently uses pytest for unittesting the routines.
To run all of the tests, just run `pytest` in the root directory. Review
the pytest [documentation](https://docs.pytest.org/en/stable/) for the full
suite of what you can do with it.

## Installation

To install the tritonblas library, just run `pip install .` in the root directory.

## Development

### Adding a new routine

#### Creating the kernel and wrapper function

Create a python file in the level directory matching the routine you want to add.
Implement the Triton kernel and write a wrapper function that can be called
externally by users of the routine.

#### Exposing the wrapper function for external use

To expose the new routine, add an import to the `tritonblas/__init__.py` file.

```python
from .level3.routine_filename import wrapper
```

This will enable users to call your routine by
`output = tritonblas.wrapper(inputs)`.

#### Creating unittests for the new routine

Create a `test_routine.py` under the `tests` directory. An example boilerplate
for a test could look like this:

```python
import tritonblas as tb

def test_routine():
    input = test_values

    output = tb.routine(input)

    triton.testing.assert_close(output, expected_output)
```

Additional tests can be added to the file to validate different usages of the
new routine.

#### Creating benchmarks for the new routine

For benchmarks, that is still a work in progress. Currently, pytest-benchmark
can be used but there is no integration between the `triton.testing.Benchmark`
and pytest-benchmark.

### Making use of the library

To use a routine from the library is as simple as:

```python
import tritonblas as tb

output = tb.routine(inputs)
```
