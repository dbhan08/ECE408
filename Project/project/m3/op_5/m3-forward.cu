#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const __half * input, __half *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
    const size_t W_unroll = Batch * Height_out * Width_out; // Unrolled width
    const size_t H_unroll = Channel * K * K;                // Unrolled height

    #define in_4d(b, c, h, w) input[((b) * Channel + (c)) * Height * Width + (h) * Width + (w)]
    #define out_2d(h, w) output[(h) * W_unroll + (w)]

    size_t h_unroll = blockIdx.y * blockDim.y + threadIdx.y;
    size_t w_unroll = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_unroll < H_unroll && w_unroll < W_unroll) {
        int c = h_unroll / (K * K);
        int p = (h_unroll % (K * K)) / K;
        int q = h_unroll % K;
        int b = w_unroll / (Height_out * Width_out);
        int s = w_unroll % (Height_out * Width_out);
        int h_out = s / Width_out;
        int w_out = s % Width_out;

        int h_in = h_out + p;
        int w_in = w_out + q;

        // Boundary check for input dimensions
        if (h_in < Height && w_in < Width) {
            out_2d(h_unroll, w_unroll) = in_4d(b, c, h_in, w_in);
        } else {
            out_2d(h_unroll, w_unroll) = __float2half(0.0f);
        }
    }

    #undef in_4d
    #undef out_2d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const __half * A, const __half *B, __half *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ __half tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    __half val = __float2half(0.0f);

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = __float2half(0.0f);
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = __float2half(0.0f);
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val = __hadd(val, __hmul(tileA[ty][i], tileB[i][tx]));

            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const __half *input, __half *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    size_t input_size = Batch * Channel * Height * Width;
    size_t mask_size = Map_out * Channel * K * K;
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);

    cudaMalloc((void**)device_input_ptr, input_size * sizeof(__half));
    cudaMalloc((void**)device_mask_ptr, mask_size*sizeof(__half));
    cudaMalloc((void**)device_output_ptr, output_size*sizeof(__half));

    __half *host_input_half = (__half *)malloc(input_size * sizeof(__half));
    __half *host_mask_half = (__half *)malloc(mask_size * sizeof(__half));

    for (size_t i = 0; i < input_size; ++i) {
        host_input_half[i] = __float2half(host_input[i]);
    }
    for (size_t i = 0; i < mask_size; ++i) {
        host_mask_half[i] = __float2half(host_mask[i]);
    }

    cudaMemcpy(*device_input_ptr, host_input_half, input_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask_half, mask_size * sizeof(__half), cudaMemcpyHostToDevice);
    
    free(host_input_half);
    free(host_mask_half);


    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float * device_output, const float * device_input, const float  *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t Width_unrolled = Batch * (Height - K + 1) * (Width - K + 1);
    size_t Height_unrolled = Channel * K * K;
    const int W_unroll = Batch * Height_out * Width_out; // Unrolled width
    const int H_unroll = Channel * K * K; 
  

    __half *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    __half *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(__half));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(__half));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    int num_tile_h = (Height_out - 1) / TILE_WIDTH + 1;
    int num_tile_w = (Width_out - 1) / TILE_WIDTH + 1;
    int total_tiles = num_tile_h * num_tile_w;

    dim3 block_unroll_dim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid_unroll_dim((W_unroll + TILE_WIDTH - 1) / TILE_WIDTH,
                     (H_unroll + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    matrix_unrolling_kernel<<<grid_unroll_dim, block_unroll_dim>>>(
       (const __half *) device_input, unrolled_matrix, Batch, Channel, Height, Width, K
    );
    // TODO: Set the kernel dimensions and call the matmul kernel
    int numARows = Map_out;
    int numAColumns = Channel * K * K;
    int numBRows = numAColumns;
    int numBColumns = Width_unrolled;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    dim3 grid_matmul_dim((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
                         (numCRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 block_matmul_dim(TILE_WIDTH, TILE_WIDTH, 1);

    matrixMultiplyShared<<<grid_matmul_dim, block_matmul_dim>>>(
         (const __half *) device_mask, unrolled_matrix, matmul_output,
        numARows, numAColumns,
        numBRows, numBColumns,
        numCRows, numCColumns
    );

    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output,  (__half *) device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    __half *host_output_half = (__half *)malloc(output_size * sizeof(__half));

    cudaMemcpy(host_output_half, device_output, output_size *sizeof(__half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < output_size; ++i) {
        host_output[i] = __half2float(host_output_half[i]);
    }

    // Free temporary host __half array
    free(host_output_half);


    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}