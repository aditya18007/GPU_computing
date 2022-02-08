// Implement kernels here (Note: delete sample code below)
#include <stdio.h>
#include <iostream>
#include "ahe_gpu.h"
#include "utils.h"
#include<map>
__constant__ int width[1];
__constant__ int heigth[1];

__constant__ int ntiles_x[1];
__constant__ int ntiles_y[1];

__global__ void ahe_GPU1(unsigned char* img_in, unsigned char* mappings)
{
    int x = (blockDim.x*blockIdx.x + threadIdx.x)*TILE_SIZE_X;
    int y = (blockDim.y*blockIdx.y + threadIdx.y)*TILE_SIZE_Y;
    if ( x < width[0] && y < heigth[0]){

        int pdf[256], cdf[256];
            
        for(int i = 0; i < 256; i++){
            pdf[i] = 0;
        }
            
        for(int j=y; j<(y+TILE_SIZE_Y); j++) {
		    for(int i=x; i<(x+TILE_SIZE_X); i++){
				pdf[img_in[i+j*width[0]]]++;
			}
		}
        
        cdf[0] = pdf[0];
		for(int i=1; i< 256; i++){
			cdf[i] = cdf[i-1] + pdf[i];
		}

        int cdf_min = PIXELS_PER_TILE+1; // minimum non-zero value of the CDF
		for(int i=0; i<256; i++){
			if(cdf[i] != 0) {
				cdf_min = cdf[i]; 
				break;
			}
		}

        int tile_i = x / TILE_SIZE_X;
		int tile_j = y / TILE_SIZE_Y;
		int offset = 256*(tile_i + tile_j*ntiles_x[0]);
        for(int i=0; i< 256; i++){
			mappings[i+ offset] = (unsigned char)round(255.0 * float(cdf[i] - cdf_min)/float(PIXELS_PER_TILE - cdf_min));
		}
                
    }
}

void get_mappings( unsigned char* d_img_in, unsigned char* d_mappings, int width_, int height_){
    int num_threads_x = 16;
    int num_threads_y = 16;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);  

    int num_blocks_x = (width_ / TILE_SIZE_X) + 1; 
    int num_blocks_y = (height_ / TILE_SIZE_Y) + 1;
    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1); 

    ahe_GPU1<<< grid_shape, block_shape>>>(d_img_in, d_mappings);
}

void print_mappings(unsigned char* d_mappings, int mapping_size){
    unsigned char* mappings = new unsigned char[mapping_size];
    int mapping_size_bytes = mapping_size*sizeof(unsigned char);
    SAFE_CALL(cudaMemcpy, (void*)mappings, (void*)d_mappings,mapping_size_bytes, cudaMemcpyDeviceToHost)
    
    std::map<int, int> counts;
    std::cout << "Printing mappings on GPU\n";
    for(int i = 0; i < mapping_size; i++){
        int i__ = mappings[i];
        if(counts.find(i__) == counts.end()){
            counts[i__] = 1;
            continue;
        }
        counts[i__]++;
    }
    for(auto& p : counts){
        std::cout << p.first << ':' << p.second << '\n';
    }
    std::cout << std::endl;
    delete []mappings;   
}
extern "C" void run_ahe_GPU(unsigned char* img_in, unsigned char* img_out, int width_, int height_){
    
	int ntiles_x_ = width_ / TILE_SIZE_X;
	int ntiles_y_ = height_ / TILE_SIZE_Y;

    SAFE_CALL( cudaMemcpyToSymbol, width, &width_, sizeof(width_))
    SAFE_CALL( cudaMemcpyToSymbol, heigth, &height_, sizeof(height_))
    SAFE_CALL( cudaMemcpyToSymbol, ntiles_x, &ntiles_x_, sizeof(ntiles_x_))
    SAFE_CALL( cudaMemcpyToSymbol, ntiles_y, &ntiles_y_, sizeof(ntiles_y_))

    auto img_size_bytes = height_*width_*sizeof(unsigned char);
    
    unsigned char *d_img_in;
    SAFE_CALL( cudaMalloc, (void**)&d_img_in, img_size_bytes)
    SAFE_CALL( cudaMemcpy, (void*)d_img_in, (void*)img_in,img_size_bytes, cudaMemcpyHostToDevice)

    unsigned char *d_img_out;
    SAFE_CALL( cudaMalloc, (void**)&d_img_out, img_size_bytes)

    unsigned char *d_mappings;
    auto mapping_size = ntiles_x_*ntiles_y_*256;
    auto mapping_size_bytes = mapping_size*sizeof(unsigned char);
    SAFE_CALL( cudaMalloc, (void**)&d_mappings, mapping_size_bytes)

    SAFE_CALL(cudaDeviceSynchronize)
    print_mappings(d_mappings, mapping_size);
    SAFE_CALL( cudaFree, d_img_in)
    SAFE_CALL( cudaFree, d_mappings)
    SAFE_CALL( cudaFree, d_img_out)
    
}