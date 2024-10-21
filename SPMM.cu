#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void spmm_coo_kernel(int *row, int *col, float *val, 
                            float *dense_matrix, float *out_matrix,
                            int nnz, int num_cols, int num_rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nnz){
        int r = row[tid];
        int c = col[tid];
        float v = val[tid];

        for (int i = 0; i < num_cols; i++){
            atomicAdd(
                &out_matrix[r * num_cols + i],
                v * dense_matrix[c * num_cols + i]
            );
        }
    }
}

void spmm_coo(int *row, int *col, float *val, float *dense_matrix,
                float *out_matrix, int nnz, int num_cols, int num_rows) {
    int *d_row, *d_col;
    float *d_val, *d_dense_matrix, *d_out_matrix;

    cudaMalloc((void **)&d_row, nnz * sizeof(int));
    cudaMalloc((void **)&d_col, nnz * sizeof(int));
    cudaMalloc((void **)&d_val, nnz * sizeof(float));
    cudaMalloc((void **)&d_dense_matrix, num_cols * num_rows * sizeof(float));
    cudaMalloc((void **)&d_out_matrix, num_cols * num_rows * sizeof(float));

    cudaMemcpy(d_row, row, nnz *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, nnz *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nnz *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense_matrix, dense_matrix,
               num_cols * num_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out_matrix, 0, num_cols * num_rows * sizeof(float));

    int num_blocks = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmm_coo_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_row, d_col, d_val, d_dense_matrix, d_out_matrix, nnz, num_cols, num_rows
    );

    cudaMemcpy(out_matrix, d_out_matrix,
               num_cols * num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_dense_matrix);
    cudaFree(d_out_matrix);
}