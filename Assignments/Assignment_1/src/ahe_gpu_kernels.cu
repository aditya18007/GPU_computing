// Implement kernels here (Note: delete sample code below)
#include <stdio.h>
#include <iostream>
#include "ahe_gpu.h"
#include "utils.h"

__constant__ int width;
__constant__ int heigth;

__constant__ int pixels_per_tile;
__constant__ int ntiles_x;
__constant__ int ntiles_y;

__global__ void ahe_GPU(unsigned char* img_out)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    img_out[x+y*width] = 255;
}

extern "C" void run_ahe_GPU(unsigned char* img_in, unsigned char* img_out, int width_, int height_){
    int pixels_per_tile_ = TILE_SIZE_X*TILE_SIZE_Y;
	int ntiles_x_ = width_ / TILE_SIZE_X;
	int ntiles_y_ = height_ / TILE_SIZE_Y;

    SAFE_CALL( cudaMemcpyToSymbol, &width, &width_, sizeof(width_))
    SAFE_CALL( cudaMemcpyToSymbol, &heigth, &height_, sizeof(height_))
    SAFE_CALL( cudaMemcpyToSymbol, &pixels_per_tile, &pixels_per_tile_, sizeof(pixels_per_tile_))
    SAFE_CALL( cudaMemcpyToSymbol, &ntiles_x, &ntiles_x_, sizeof(width))
    SAFE_CALL( cudaMemcpyToSymbol, &ntiles_y, &ntiles_y_, sizeof(width))

    int num_threads_x = 16;
    int num_threads_y = 16;
    dim3 block_shape = dim3( num_threads_x, num_threads_y );  

    int num_blocks_x = (width_ / num_threads_x) + 1; 
    int num_blocks_y = (height_ / num_threads_y) + 1;

    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y ); 

    unsigned char* d_img_out;
    auto img_size_bytes = height_*width_*sizeof(unsigned char);
    SAFE_CALL( cudaMalloc, &d_img_out, img_size_bytes)

    ahe_GPU<<< grid_shape, block_shape>>>(d_img_out);
    cudaDeviceSynchronize();
    SAFE_CALL( cudaMemcpy, img_out, d_img_out, img_size_bytes, cudaMemcpyHostToDevice )

}