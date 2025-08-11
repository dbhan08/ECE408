// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < len) {
    out[j] = in1[j] + in2[j];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);
  int size = inputLength * sizeof(float);
  float *a, *b, *c;
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &a, size);
  cudaMalloc((void**) &b, size);
  cudaMalloc((void**) &c, size);


  //@@ Copy memory to the GPU here
  cudaMemcpy(a,hostInput1,size,cudaMemcpyHostToDevice);
  cudaMemcpy(b,hostInput2,size,cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(256,1,1);
  int grid = inputLength/256;
  if(inputLength % 256 != 0) {
    grid++;
  }
  dim3 DimGrid(grid,1,1);
  

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(a,b,c,inputLength);


  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput,c,size,cudaMemcpyDeviceToHost);



  //@@ Free the GPU memory here
  cudaFree(a); cudaFree(b); cudaFree(c);


  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
