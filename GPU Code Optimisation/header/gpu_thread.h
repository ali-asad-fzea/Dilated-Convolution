#include "cuda.h"
 
#define _2D_BLOCK_X 32
#define _2D_BLOCK_Y 32
 
__global__ void KernelConv
(
    int input_row,
    int input_col,
    int kernel_row,
    int kernel_col,
    int output_row,
    int output_col,
    int *input,
    int *kernel,
    long long unsigned int *output
)
{
    size_t col_output = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_output = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (row_output >= output_row || col_output >= output_col) return;
 
    long long unsigned int partialSum = 0;
    int row, col;
    for (int i = 0; i < kernel_row; i++){
         for (int j = 0; j < kernel_col; j++){
            row = (row_output + (i << 1)) % input_row;
            col = (col_output + (j << 1)) % input_col;
            partialSum += input[row * input_col + col] * kernel[i * kernel_col + j];
        }
    }
    output[row_output * output_col + col_output] = partialSum;
}
 
void gpuThread(
    int input_row,
    int input_col,
    int *input,
    int kernel_row,
    int kernel_col,
    int *kernel,
    int output_row,
    int output_col,
    long long unsigned int *output
) {
    dim3 blocks(_2D_BLOCK_X, _2D_BLOCK_Y);
    size_t grid_col = std::ceil((double)output_col / blocks.x);
    size_t grid_row = std::ceil((double)output_row / blocks.y);
    dim3 grid(grid_col, grid_row);
 
    int *input_g, *out_k;
    long long unsigned int *out_g;
 
    cudaMalloc(&out_k, kernel_row * kernel_col * sizeof(int));
    cudaMalloc(&input_g, input_row * input_col * sizeof(int));
    cudaMalloc(&out_g, output_row * output_col * sizeof(long long unsigned int));
    cudaMemcpy(input_g, input, input_row * input_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_k, kernel,  kernel_row * kernel_col * sizeof(int), cudaMemcpyHostToDevice);
 
    KernelConv<<<grid, blocks>>>(
        input_row,
        input_col,
        kernel_row,
        kernel_col,
        output_row,
        output_col,
        input_g,
        out_k,
        out_g
    );
 
    cudaMemcpy(output, out_g,  sizeof(long long unsigned int) * output_row * output_col, cudaMemcpyDeviceToHost);
    cudaFree(input_g);
    cudaFree(out_g);
    cudaFree(out_k);
}