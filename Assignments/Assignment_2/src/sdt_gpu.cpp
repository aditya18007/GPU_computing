#include <cuda_runtime.h>
#include <iostream>

#include "sdt_gpu.h"

extern "C" void gpu_main(unsigned char* bitmap_raw, float *sdt, int width, int height);

void computeSDT_GPU(unsigned char * bitmap_raw, float *sdt, int width, int height)
{
    size_t sz = width*height;
    gpu_main(bitmap_raw, sdt, width, height);
}

