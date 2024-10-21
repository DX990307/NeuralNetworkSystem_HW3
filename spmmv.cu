#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void spmm_coo_kernel_v(int *row, int *col, float *val, 
                                  float *dense_matrices, float *out_matrices,
                                  int nnz, int num_cols, int num_rows, int num_matrices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nnz) {
        int r = row[tid];
        int c = col[tid];
        float v = val[tid];

        for (int m = 0; m < num_matrices; m++) {
            for (int i = 0; i < num_cols; i++) {
                atomicAdd(
                    &out_matrices[m * num_rows * num_cols + r * num_cols + i],
                    v * dense_matrices[m * num_rows * num_cols + c * num_cols + i]
                );
            }
        }
    }
}

void spmm_coo_v(int *row, int *col, float *val, float *dense_matrices,
                float *out_matrices, int nnz, int num_cols, int num_rows, int num_matrices) {
    int *d_row, *d_col;
    float *d_val, *d_dense_matrices, *d_out_matrices;

    cudaMalloc((void **)&d_row, nnz * sizeof(int));
    cudaMalloc((void **)&d_col, nnz * sizeof(int));
    cudaMalloc((void **)&d_val, nnz * sizeof(float));
    cudaMalloc((void **)&d_dense_matrices, num_matrices * num_cols * num_rows * sizeof(float));
    cudaMalloc((void **)&d_out_matrices, num_matrices * num_cols * num_rows * sizeof(float));

    cudaMemcpy(d_row, row, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense_matrices, dense_matrices,
               num_matrices * num_cols * num_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out_matrices, 0, num_matrices * num_cols * num_rows * sizeof(float));

    int num_blocks = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmm_coo_kernel_v<<<num_blocks, BLOCK_SIZE>>>(
        d_row, d_col, d_val, d_dense_matrices, d_out_matrices, nnz, num_cols, num_rows, num_matrices
    );

    cudaMemcpy(out_matrices, d_out_matrices,
               num_matrices * num_cols * num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_dense_matrices);
    cudaFree(d_out_matrices);
}