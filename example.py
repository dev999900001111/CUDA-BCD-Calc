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
