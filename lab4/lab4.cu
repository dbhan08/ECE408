#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8

//@@ Define constant memory for device kernel here
__constant__ float Minecraft[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  int r = MASK_WIDTH / 2;

  int x = blockIdx.x * TILE_WIDTH + threadIdx.x - r;
  int y = blockIdx.y * TILE_WIDTH + threadIdx.y - r;
  int z = blockIdx.z * TILE_WIDTH + threadIdx.z - r;

  __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1]
                       [TILE_WIDTH + MASK_WIDTH - 1]
                       [TILE_WIDTH + MASK_WIDTH - 1];

  if (x >= 0 && x < x_size &&
      y >= 0 && y < y_size &&
      z >= 0 && z < z_size) {
    N_ds[threadIdx.z][threadIdx.y][threadIdx.x] = input[z * y_size * x_size + y * x_size + x];
  } else {
    N_ds[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  if (threadIdx.x >= r && threadIdx.x < TILE_WIDTH + r &&
      threadIdx.y >= r && threadIdx.y < TILE_WIDTH + r &&
      threadIdx.z >= r && threadIdx.z < TILE_WIDTH + r) {

    float Pvalue = 0.0f;

    for (int i = 0; i < MASK_WIDTH; i++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        for (int k = 0; k < MASK_WIDTH; k++) {
          Pvalue += Minecraft[i][j][k] * N_ds[threadIdx.z - r + i][threadIdx.y - r + j][threadIdx.x - r + k];
        }
      }
    }

    int output_x = x;
    int output_y = y;
    int output_z = z;

    if (output_x >= 0 && output_x < x_size &&
        output_y >= 0 && output_y < y_size &&
        output_z >= 0 && output_z < z_size) {
      output[output_z * y_size * x_size + output_y * x_size + output_x] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions

  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));



  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Minecraft, hostKernel, kernelLength * sizeof(float));




  //@@ Initialize grid and block dimensions here

  dim3 dimBlock(TILE_WIDTH + (MASK_WIDTH-1),TILE_WIDTH + (MASK_WIDTH-1) , TILE_WIDTH + (MASK_WIDTH-1));
  dim3 dimGrid(ceil(x_size/(1.0*TILE_WIDTH)), ceil(y_size/(1.0*TILE_WIDTH)), ceil(z_size/(1.0*TILE_WIDTH)));

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);


  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);





  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

