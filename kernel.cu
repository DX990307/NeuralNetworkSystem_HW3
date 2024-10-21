#include "kernel.h"

__global__ void spmmv_kernel(
    int vcount,
    int feature_dim,
    int* offsets,
    int* neighbors,
    float* input_data,
    float* output_data,
    bool norm
) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (src >= vcount || feat_idx >= feature_dim)
        return;

    float sum = 0.0f;
    int row_start = offsets[src];
    int row_end = offsets[src + 1];
    int degree = row_end - row_start;

    for (int offset = row_start; offset < row_end; ++offset) {
        int dst = neighbors[offset];
        sum += input_data[dst * feature_dim + feat_idx];
    }

    if (norm && degree > 0) {
        sum /= degree;
    }

    output_data[src * feature_dim + feat_idx] = sum;
}

void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm) {
    int vcount = graph.get_vcount();
    int ecount = graph.get_ecount();
    int feature_dim = input1.col_count;

    // Get offsets and neighbors from the graph
    int* h_offsets = nullptr;
    int* h_neighbors = nullptr;

    if (!reverse) {
        h_offsets = graph.offset_csr;
        h_neighbors = graph.nebrs_csr;
    } else {
        h_offsets = graph.offset_csc;
        h_neighbors = graph.nebrs_csc;
    }

    // Allocate device memory for offsets and neighbors
    int* d_offsets;
    int* d_neighbors;

    int num_offsets = vcount + 1;
    cudaMalloc(&d_offsets, sizeof(int) * num_offsets);
    cudaMalloc(&d_neighbors, sizeof(int) * ecount);

    // Copy offsets and neighbors to device memory
    cudaMemcpy(d_offsets, h_offsets, sizeof(int) * num_offsets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, h_neighbors, sizeof(int) * ecount, cudaMemcpyHostToDevice);

    // Launch kernel with adjusted block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((vcount + blockSize.x - 1) / blockSize.x, (feature_dim + blockSize.y - 1) / blockSize.y);

    spmmv_kernel<<<gridSize, blockSize>>>(
        vcount,
        feature_dim,
        d_offsets,
        d_neighbors,
        input1.data_ptr,
        output.data_ptr,
        norm
    );

    // Free device memory
    cudaFree(d_offsets);
    cudaFree(d_neighbors);
}
