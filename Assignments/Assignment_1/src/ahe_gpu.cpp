#include "ahe_gpu.h"
#include <cuda_runtime.h>

#include <iostream>

extern "C" void run_ahe_GPU(unsigned char* img_in, unsigned char* img_out, int width, int height);

void adaptiveEqualizationGPU(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  run_ahe_GPU(img_in,img_out, width, height);
}
