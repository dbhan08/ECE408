#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void fusion_kernel(const float *input, const float *mask, float *output,
                                   const int Batch, const int Map_out, const int Channel,
                                   const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int W_unroll = Batch * Height_out * Width_out;
    const int H_unroll = Channel * K * K;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;

    float output_val = 0.0f;

    for (int tileId = 0; tileId < (H_unroll - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < Map_out && tileId * TILE_WIDTH + tx < H_unroll) {
            tileA[ty][tx] = mask[row * H_unroll + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (col < W_unroll && tileId * TILE_WIDTH + ty < H_unroll) {
            int c = (tileId * TILE_WIDTH + ty) / (K * K);
            int p = ((tileId * TILE_WIDTH + ty) % (K * K)) / K;
            int q = (tileId * TILE_WIDTH + ty) % K;

            int b = col / (Height_out * Width_out);
            int s = col % (Height_out * Width_out);
            int h_out = s / Width_out;
            int w_out = s % Width_out;

            int h_in = h_out + p;
            int w_in = w_out + q;

            if (h_in < Height && w_in < Width) {
                tileB[ty][tx] = input[((b) * Channel + (c)) * Height * Width + (h_in) * Width + (w_in)];
            } else {
                tileB[ty][tx] = 0.0f;
            }
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < Map_out && col < W_unroll) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                if (i + 0 < TILE_WIDTH) output_val += tileA[ty][i + 0] * tileB[i + 0][tx];
              
            }
        }

        __syncthreads();
    }

    if (row < Map_out && col < W_unroll) {
        int b = col / (Height_out * Width_out);
        int s = col % (Height_out * Width_out);
        output[b * Map_out * (Height_out * Width_out) + row * (Height_out * Width_out) + s] = output_val;
    }
}

__global__ void matrix_unrolling_kernel(const float *input, float *output,
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
            out_2d(h_unroll, w_unroll) = 0;
        }
    }

    #undef in_4d
    #undef out_2d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
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
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
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
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

    cudaMalloc((void**)device_input_ptr, input_size);
    cudaMalloc((void**)device_mask_ptr, mask_size);
    cudaMalloc((void**)device_output_ptr, output_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    


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


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_unrolled = Batch * Height_out * Width_out;


    // Define grid and block dimensions
    dim3 grid_dim((Width_unrolled + TILE_WIDTH - 1) / TILE_WIDTH, 
                  (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH,1);

    // Calculate shared memory size
    size_t shared_mem_size = Channel * K * K * sizeof(float);

    // Launch the fused convolution kernel
    fusion_kernel<<<grid_dim, block_dim>>>(
        device_input, device_mask, device_output,
        Batch, Map_out, Channel, Height, Width, K
    );
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);


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