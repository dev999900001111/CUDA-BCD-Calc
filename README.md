# BCD Calculation Library for CUDA

This library provides functions for performing decimal calculations on NVIDIA GPUs using CUDA, similar to COBOL's PACKED-DECIMAL format. It aims to achieve high-precision decimal arithmetic with up to 36 significant digits.

The library uses a custom representation of decimal numbers, simulating BCD (Binary-Coded Decimal) using four int32 values. Each int32 value represents a block of 9 decimal digits, allowing for efficient storage and computation on the GPU.

Please note that this library is a work in progress and may have limitations. Currently, overflow and zero division handling are not fully implemented, but they are planned for future updates. We appreciate your understanding and patience as we continue to improve the library.

## Features

- Addition, subtraction, multiplication, and division operations for BCD numbers
- Scaling and alignment of BCD numbers
- Rounding and truncation operations
- Conversion between BCD and integer representations
- Comparison operations for BCD numbers

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.x
- NumPy
- Numba
- CuPy

## Installation

1. Install the required dependencies:
   ```
   pip install numpy numba cupy
   ```

2. Clone the repository or download the library files.

3. Import the library in your Python code:
   ```python
   from numba_bcdlike import *
   ```

## Usage

The library uses a specific representation for decimal numbers as int32 arrays. Each array consists of 7 elements:
- Elements 0 to 3: The decimal digits of the number, where each element represents a block of 9 digits. The elements are ordered in a little-endian-like manner, with element 0 representing the least significant digits and element 3 representing the most significant digits.
- Element 4: The scale of the number, indicating the number of decimal places.
- Element 5: The sign of the number (-1 for negative, 1 for positive).
- Element 6: A flag indicating whether the number has a sign (1 for signed, 0 for unsigned).

Here's an example that demonstrates the use of elements 2, 3, and 4 without causing overflow:

```python
import cupy as cp
from cuda_bcd_calc import gpu_add, gpu_set_dec
from numba import cuda

@cuda.jit(cache=True)
def kernel_add(target, output):
    # Get the thread ID
    thread_id = cuda.grid(1)
    
    # Check if the thread ID is within the target array range
    if thread_id < target.shape[0]:
        # Perform addition using the gpu_add function
        gpu_add(target[thread_id][0], target[thread_id][1], output[thread_id])

@cuda.jit(cache=True)
def kernel_to_ascii(target, output):
    # Get the thread ID
    thread_id = cuda.grid(1)
    
    # Check if the thread ID is within the target array range
    if thread_id < target.shape[0]:
        # Convert the decimal array to ASCII representation using gpu_set_dec
        gpu_set_dec(output[thread_id], 0, 36, target[thread_id])

# Create decimal arrays on the GPU
a = cp.array([123456789, 987654321, 456789012, 0, 8, 1, 1], dtype=cp.int32)
b = cp.array([987654321, 123456789, 543210987, 0, 8, 1, 1], dtype=cp.int32)
d_sample = cp.array([[a, b]])

# Create an array to store the calculation results
d_result = cp.array([cp.array([0, 0, 0, 0, 8, 1, 1], dtype=cp.int32)])

# Set the grid and block sizes
threads_per_block = 256
blocks_per_grid = (d_sample.shape[0] + threads_per_block - 1) // threads_per_block

# Launch the kernel function for addition
kernel_add[blocks_per_grid, threads_per_block](d_sample, d_result)

# Create an array to store the ASCII representation
d_ascii = cp.array([cp.zeros(37, dtype=cp.int32)])

# Launch the kernel function for converting to ASCII
kernel_to_ascii[blocks_per_grid, threads_per_block](d_result, d_ascii)

# Convert the CuPy array to a string
result_string = cp.asnumpy(d_ascii).tobytes().decode('ascii')

print(result_string) # Output: +00000001000000000111111111111111110
```

In this example, `a` represents the decimal number 456789012.98765432123456789 and `b` represents the decimal number 543210987.12345678998765432. The `gpu_add` function performs the addition operation, and the result is stored in the `result` array. Finally, the `gpu_to_dec` function converts the decimal representation back to a standard decimal number.

The library's ability to handle large decimal numbers with high precision is showcased in this example. By utilizing elements 2 and 3, which represent the higher-order digits, the library can perform calculations on numbers with up to 36 significant digits.

Note that the scale (element 4) is set to 8 in this example, indicating that there are 8 decimal places in the numbers. The library automatically aligns the scale of the operands before performing the calculation to ensure accurate results.

The example avoids overflow by ensuring that the sum of the numbers does not exceed the maximum value that can be represented by the library (999999999.999999999999999999999999999).


## API Reference

The library provides the following functions:

- `gpu_add(a, b, result)`: Performs BCD addition of `a` and `b` and stores the result in `result`.
- `gpu_sub(a, b, result)`: Performs BCD subtraction of `b` from `a` and stores the result in `result`.
- `gpu_mul(a, b, result)`: Performs BCD multiplication of `a` and `b` and stores the result in `result`.
- `gpu_div(a, b, result)`: Performs BCD division of `a` by `b` and stores the result in `result`.
- `gpu_set_dec(result, scale, signed, value)`: Converts a BCD array to its ASCII representation.
- `gpu_to_dec(bcd_array)`: Converts a BCD array to its decimal representation.
- `gpu_from_dec(decimal_value, scale, signed)`: Converts a decimal value to its BCD array representation with the specified scale and sign.
- `gpu_align_scale(a, b, result_a, result_b)`: Aligns the scale of two BCD arrays `a` and `b` and stores the aligned arrays in `result_a` and `result_b`.
- `gpu_shift_left(bcd_array, count)`: Shifts the BCD array to the left by the specified `count` of digits.
- `gpu_shift_right_round(bcd_array, count)`: Shifts the BCD array to the right by the specified `count` of digits with rounding.
- `gpu_shift_right_ceil(bcd_array, count)`: Shifts the BCD array to the right by the specified `count` of digits with ceiling rounding.
- `gpu_shift_right_floor(bcd_array, count)`: Shifts the BCD array to the right by the specified `count` of digits with floor rounding.
- `gpu_is_zero(bcd_array)`: Checks if the BCD array represents zero.
- `gpu_compare(a, b)`: Compares two BCD arrays `a` and `b`. Returns 1 if `a` is greater, -1 if `b` is greater, and 0 if they are equal.

## License

This library is released under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Acknowledgements

This library was inspired by the need for efficient BCD calculations on GPUs, similar to those performed in COBOL's PACKED-DECIMAL format.

## Future Plans

We are constantly working to improve the BCD Calculation Library for CUDA, and we have several enhancements planned for future releases:

- **Memory Optimization**: We plan to compact metadata into a single int32 value to reduce memory consumption significantly. This change will optimize the storage format and improve performance, especially for large-scale operations on GPUs.
- **Error Handling Improvements**: Comprehensive overflow and zero division handling will be implemented to ensure robustness and reliability of decimal calculations.
- **Additional Mathematical Functions**: Expansion of the library to include more complex mathematical functions such as power, square root, and logarithmic calculations in the BCD format.
- **Performance Enhancements**: Ongoing optimizations to further leverage CUDA capabilities for faster processing and lower latency in high-volume computations.

Please stay tuned for updates as we continue to expand the library's capabilities and enhance its performance.
