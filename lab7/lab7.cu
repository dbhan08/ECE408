// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define R_WEIGHT 0.21f
#define G_WEIGHT 0.71f
#define B_WEIGHT 0.07f
#define blockSize 32


//@@ insert code here
__global__ void floatToUcharKernel(float *inputImage, unsigned char *outputImage, int imageSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (idx < imageSize) {
        float inputValue = inputImage[idx];

        unsigned char outputValue = (inputValue >= 1.0f) ? 255 :(inputValue <= 0.0f) ? 0 :(unsigned char)(inputValue * 255);

        outputImage[idx] = outputValue;
    }
}

__global__ void rgbToGrayscaleKernel(unsigned char *inputImage, unsigned char *grayImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;       
        int rgbIdx = idx * 3;          

        grayImage[idx] = (unsigned char)(R_WEIGHT * inputImage[rgbIdx] +G_WEIGHT * inputImage[rgbIdx + 1] + B_WEIGHT * inputImage[rgbIdx + 2]);
                                        
    }
}
__global__ void histogramKernel(unsigned char *input_data, unsigned int *global_histogram, int data_size) {
    __shared__ unsigned int block_histogram[HISTOGRAM_LENGTH]; 

    if (threadIdx.x < HISTOGRAM_LENGTH) {
        block_histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    while (global_index < data_size) {
        atomicAdd(&(block_histogram[input_data[global_index]]), 1);  
        global_index += total_threads;  
    }
    __syncthreads();  
    if (threadIdx.x < HISTOGRAM_LENGTH) {
        atomicAdd(&(global_histogram[threadIdx.x]), block_histogram[threadIdx.x]);
    }
}

__device__ float clamp(float value, float minVal, float maxVal) {
    return min(max(value, minVal), maxVal);
}

__device__ float correct_color(float pixelValue, float cdfValue, float cdfMin) {
    return (HISTOGRAM_LENGTH - 1) * (cdfValue - cdfMin) / (1.0f - cdfMin);
}

__global__ void equalizeKernel(unsigned char *pixelValues, float *cdf, int numPixels) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPixels) {
        float equalizedValue = correct_color(pixelValues[idx], cdf[pixelValues[idx]], cdf[0]);
        pixelValues[idx] = (unsigned char)clamp(equalizedValue, 0.0f, (float)(HISTOGRAM_LENGTH - 1));
    }
}

__global__ void ucharToFloatKernel(unsigned char *ucharImage, float *outputImage, int imageSize) {
    int bd = blockDim.x;
    int bI = blockIdx.x;
    int tx = threadIdx.x;
    int idx = bI * bd + tx;

    if (idx < imageSize) {
        outputImage[idx] = ucharImage[idx] / 255.0f;
    }
}


void calculateCDFSequentially(unsigned int *histogram, float *cdf, int width, int height) {
   int totalPixels = width * height;
   cdf[0] = (float)histogram[0] / totalPixels;
   for (int i = 1; i < HISTOGRAM_LENGTH; ++i) {
       cdf[i] = cdf[i - 1] + (float)histogram[i] / totalPixels;
   }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *d_inputImageData;
  unsigned char *d_imageCharData;
  unsigned char *d_imageGrayscaleData;
  unsigned int *d_histogramData;
  float *d_CDFData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  // Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  //@@ Insert code here
  int totalPixels = imageWidth * imageHeight;
  int totalImageSize = totalPixels * imageChannels * sizeof(float);
  int grayscaleSize = totalPixels * sizeof(unsigned char);
  int histogramSize = HISTOGRAM_LENGTH * sizeof(unsigned int);
  int CDFSize = HISTOGRAM_LENGTH * sizeof(float);

  cudaMalloc((void **)&d_inputImageData, totalImageSize);
  cudaMalloc((void **)&d_imageCharData, totalImageSize / sizeof(float) * sizeof(unsigned char));
  cudaMalloc((void **)&d_imageGrayscaleData, grayscaleSize);
  cudaMalloc((void **)&d_histogramData, histogramSize);
  cudaMalloc((void **)&d_CDFData, CDFSize);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  cudaMemcpy(d_inputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  int dimBlockSize = blockSize * blockSize;
  int totalImageSizeChannels = imageWidth * imageHeight * imageChannels;
  totalImageSize = imageWidth * imageHeight;

  dim3 dimBlock_floatToUchar(dimBlockSize, 1, 1);
  dim3 dimGrid_floatToUchar((totalImageSizeChannels + dimBlockSize - 1) / dimBlockSize, 1, 1);
  floatToUcharKernel<<<dimGrid_floatToUchar, dimBlock_floatToUchar>>>(d_inputImageData, d_imageCharData, totalImageSizeChannels);

  int dimBlockSize3 = blockSize;  
  dim3 dimBlock_rgbToGrayscale(dimBlockSize3, dimBlockSize3, 1);
  dim3 dimGrid_rgbToGrayscale((imageWidth + dimBlockSize3 - 1) / dimBlockSize3, (imageHeight + dimBlockSize3 - 1) / dimBlockSize3, 1);
  rgbToGrayscaleKernel<<<dimGrid_rgbToGrayscale, dimBlock_rgbToGrayscale>>>(d_imageCharData, d_imageGrayscaleData, imageWidth, imageHeight);


  dim3 dimBlock_histogram(dimBlockSize, 1, 1);
  dim3 dimGrid_histogram((totalImageSize + dimBlockSize - 1) / dimBlockSize, 1, 1);
  histogramKernel<<<dimGrid_histogram, dimBlock_histogram>>>(d_imageGrayscaleData, d_histogramData, totalImageSize);

  unsigned int hostHistogram[HISTOGRAM_LENGTH];
  float hostCDF[HISTOGRAM_LENGTH];
  cudaMemcpy(hostHistogram, d_histogramData, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  calculateCDFSequentially(hostHistogram, hostCDF, imageWidth, imageHeight);
  cudaMemcpy(d_CDFData, hostCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock_equalize(dimBlockSize, 1, 1);
  dim3 dimGrid_equalize((totalImageSizeChannels + dimBlockSize - 1) / dimBlockSize, 1, 1);
  equalizeKernel<<<dimGrid_equalize, dimBlock_equalize>>>(d_imageCharData, d_CDFData, totalImageSizeChannels);

  dim3 dimBlock_ucharToFloat(dimBlockSize, 1, 1);
  dim3 dimGrid_ucharToFloat((totalImageSizeChannels + dimBlockSize - 1) / dimBlockSize, 1, 1);
  ucharToFloatKernel<<<dimGrid_ucharToFloat, dimBlock_ucharToFloat>>>(d_imageCharData, d_inputImageData, totalImageSizeChannels);

  cudaMemcpy(hostOutputImageData, d_inputImageData, totalImageSizeChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);

  //@@ Insert code here
  cudaFree(d_inputImageData);
  cudaFree(d_imageCharData);
  cudaFree(d_imageGrayscaleData);
  cudaFree(d_histogramData);
  cudaFree(d_CDFData);

  return 0;
}