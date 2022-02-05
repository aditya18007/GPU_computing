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
