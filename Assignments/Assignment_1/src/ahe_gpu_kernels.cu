// Implement kernels here (Note: delete sample code below)
#include <stdio.h>
#include <iostream>

__global__ void sample_kernel()
{
    int tx = threadIdx.x + blockDim.x*blockIdx.x;
	tx++;
}

extern "C" void run_sampleKernel()
{
    sample_kernel<<<1024, 1024>>>();
	cudaDeviceSynchronize();
}

extern "C" void run_ahe_GPU(unsigned char* img_in, unsigned char* img_out, int width, int height){
  
}