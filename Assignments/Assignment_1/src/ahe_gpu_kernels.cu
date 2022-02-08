// Implement kernels here (Note: delete sample code below)
#include <stdio.h>
#include <iostream>
#include "ahe_gpu.h"
#include "utils.h"
#include<map>
__constant__ int Width[1];
__constant__ int Heigth[1];

__constant__ int Ntiles_x[1];
__constant__ int Ntiles_y[1];


__global__ void ahe_GPU1(unsigned char* img_in, unsigned char* mappings)
{
    int x = (blockDim.x*blockIdx.x + threadIdx.x)*TILE_SIZE_X;
    int y = (blockDim.y*blockIdx.y + threadIdx.y)*TILE_SIZE_Y;
    int width = Width[0];
    if ( x < width && y < Heigth[0]){
        
        int pdf[256], cdf[256];
            
        for(int i = 0; i < 256; i++){
            pdf[i] = 0;
        }
            
        for(int j=y; j<(y+TILE_SIZE_Y); j++) {
		    for(int i=x; i<(x+TILE_SIZE_X); i++){
				pdf[img_in[i+j*width]]++;
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
		int offset = 256*(tile_i + tile_j*Ntiles_x[0]);
        for(int i=0; i< 256; i++){
			mappings[i+ offset] = (unsigned char)round(255.0 * float(cdf[i] - cdf_min)/float(PIXELS_PER_TILE - cdf_min));
		}      
    }
}

void get_mappings( unsigned char* d_img_in, unsigned char* d_mappings, int width_, int height_){
    int max_x = (width_ / TILE_SIZE_X) + 1;
    int max_y = (height_ / TILE_SIZE_Y) + 1;

    int num_threads_x = 16;
    int num_threads_y = 16;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);  

    int num_blocks_x = (max_x/num_threads_x) + 1; 
    int num_blocks_y = (max_y/num_threads_y) + 1;
    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1); 
    
    printf("Step 1 : Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    grid_shape.x, grid_shape.y, grid_shape.z, block_shape.x, block_shape.y, block_shape.z);

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

__global__ void ahe_GPU2(unsigned char* img_in, unsigned char* img_out, unsigned char* mappings){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int width = Width[0];
    if ( x < width && y < Heigth[0]){
            
            int ntiles_x = Ntiles_x[0];
            int ntiles_y = Ntiles_y[0];

            int tile_i0, tile_j0, tile_i1, tile_j1;
            tile_i0 = (x - TILE_SIZE_X/2) / TILE_SIZE_X;
			if(tile_i0 < 0) {
				tile_i0 = 0;
			}
			
			tile_j0 = (y - TILE_SIZE_Y/2) / TILE_SIZE_Y;
			if(tile_j0 < 0) {
				tile_j0 = 0;
			}

			tile_i1 = (x + TILE_SIZE_X/2) / TILE_SIZE_X;
			if(tile_i1 >= ntiles_x){
				tile_i1 = ntiles_x - 1;
			} 
			
			tile_j1 = (y + TILE_SIZE_Y/2) / TILE_SIZE_Y;
			if(tile_j1 >= ntiles_y) {
				tile_j1 = ntiles_y - 1;
			}

			// Find offsets to neighboring mappings. For no neighbors, set the nearest neighbor.
			int offset00 = 256*(tile_i0 + tile_j0*ntiles_x);
			int offset01 = 256*(tile_i0 + tile_j1*ntiles_x);
			int offset10 = 256*(tile_i1 + tile_j0*ntiles_x);
			int offset11 = 256*(tile_i1 + tile_j1*ntiles_x);

			// Compute 4 values and perform bilinear interpolation
      		unsigned char v00, v01, v10, v11;
			v00 = mappings[img_in[x+y*width] + offset00];
			v01 = mappings[img_in[x+y*width] + offset01];
			v10 = mappings[img_in[x+y*width] + offset10];
			v11 = mappings[img_in[x+y*width] + offset11];
			float x_frac = float(x - tile_i0*TILE_SIZE_X - TILE_SIZE_X/2)/float(TILE_SIZE_X);
			float y_frac = float(y - tile_j0*TILE_SIZE_Y - TILE_SIZE_Y/2)/float(TILE_SIZE_Y);
            float v0 = v00*(1 - x_frac) + v10*x_frac;
            float v1 = v01*(1 - x_frac) + v11*x_frac;
            float v = v0*(1 - y_frac) + v1*y_frac;

            if (v < 0){
                v = 0;
            }
                
            if (v > 255){
                v = 255;
            } 
      		img_out[x+y*width] = v;
    }
}

void adaptive_equalization( unsigned char* d_img_in, unsigned char* d_img_out, unsigned char* d_mappings, int width_, int height_ ){
    
    int num_threads_x = 16;
    int num_threads_y = 16;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);  

    int num_blocks_x = (width_ / num_threads_x) + 1; 
    int num_blocks_y = (height_ / num_threads_y) + 1;

    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1); 

    printf("Step 2 : Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    grid_shape.x, grid_shape.y, grid_shape.z, block_shape.x, block_shape.y, block_shape.z);
    
    ahe_GPU2<<< grid_shape, block_shape>>>(d_img_in, d_img_out, d_mappings);
}


extern "C" void run_ahe_GPU(unsigned char* img_in, unsigned char* img_out, int width_, int height_){
    
	int ntiles_x_ = width_ / TILE_SIZE_X;
	int ntiles_y_ = height_ / TILE_SIZE_Y;

    SAFE_CALL( cudaMemcpyToSymbol, Width, &width_, sizeof(width_))
    SAFE_CALL( cudaMemcpyToSymbol, Heigth, &height_, sizeof(height_))
    SAFE_CALL( cudaMemcpyToSymbol, Ntiles_x, &ntiles_x_, sizeof(ntiles_x_))
    SAFE_CALL( cudaMemcpyToSymbol, Ntiles_y, &ntiles_y_, sizeof(ntiles_y_))

    auto img_size_bytes = height_*width_*sizeof(unsigned char);
    
    unsigned char *d_img_in;
    SAFE_CALL( cudaMalloc, (void**)&d_img_in, img_size_bytes)
    SAFE_CALL( cudaMemcpy, (void*)d_img_in, (void*)img_in,img_size_bytes, cudaMemcpyHostToDevice)

    

    unsigned char *d_mappings;
    auto mapping_size = ntiles_x_*ntiles_y_*256;
    auto mapping_size_bytes = mapping_size*sizeof(unsigned char);
    SAFE_CALL( cudaMalloc, (void**)&d_mappings, mapping_size_bytes)
    get_mappings(d_img_in, d_mappings, width_, height_);
    SAFE_CALL(cudaDeviceSynchronize)
    
    unsigned char *d_img_out;
    SAFE_CALL( cudaMalloc, (void**)&d_img_out, img_size_bytes)
    adaptive_equalization(d_img_in, d_img_out, d_mappings, width_, height_);
    SAFE_CALL(cudaDeviceSynchronize)

    SAFE_CALL(cudaMemcpy, (void*)img_out, (void*)d_img_out,img_size_bytes, cudaMemcpyDeviceToHost)
   
    SAFE_CALL( cudaFree, d_img_in)
    SAFE_CALL( cudaFree, d_mappings)
    SAFE_CALL( cudaFree, d_img_out) 
}