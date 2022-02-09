// Implement kernels here (Note: delete sample code below)
#include <stdio.h>
#include <iostream>
#include<fstream>
#include "ahe_gpu.h"
#include "utils.h"
#include<map>

template<typename T>
void print_mappings(T* d_mappings, int mapping_size, std::string filename){
    T* mappings = new T[mapping_size];
    int mapping_size_bytes = mapping_size*sizeof(T);
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
    std::ofstream MyFile(filename + ".txt");
	for(auto& p : counts){
		MyFile << p.first << ':' << p.second << '\n';
	}
    MyFile << std::endl;
	MyFile.close();
    delete []mappings;   
}

__constant__ int Width[1];
__constant__ int Heigth[1];

__constant__ int Ntiles_x[1];
__constant__ int Ntiles_y[1];
__constant__ int Numtiles[1];

__constant__ unsigned char Mappings[32*32*256];

__global__ void ahe_get_PDF(unsigned char* img_in, int* pdf){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int width = Width[0];
    if ( x < width && y < Heigth[0]){
        int tile_i = x / TILE_SIZE_X;
		int tile_j = y / TILE_SIZE_Y;
		int offset = 256*(tile_i + tile_j*Ntiles_x[0]);
        atomicAdd(&pdf[offset + img_in[x+y*width]], 1);
    }
}

void get_pdf( unsigned char* d_img_in, int* d_pdf, int width, int height){
    
    int num_threads_x = 32;
    int num_threads_y = 32;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);  

    int num_blocks_x = (width / num_threads_x) + 1; 
    int num_blocks_y = (height / num_threads_y) + 1;

    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1); 

    printf("\nStep 1 (Get pdf): Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    grid_shape.x, grid_shape.y, grid_shape.z, block_shape.x, block_shape.y, block_shape.z);
    
    ahe_get_PDF<<< grid_shape, block_shape>>>(d_img_in, d_pdf);
}

__global__ void ahe_get_mappings(int*pdf_global, unsigned char* mappings){
    __shared__ int pdf[256];

    int i = threadIdx.x ;
    int global_i = i + blockIdx.x*blockDim.x;
    if ( global_i < Numtiles[0]*256){
		pdf[i] = pdf_global[global_i];
        __syncthreads();
        //Very naive. Can do better ?
        int cdf_i = pdf[i];
        for(int j = 0; j < i; j++){
            cdf_i += pdf[j];
        }
        int cdf_min = PIXELS_PER_TILE+1; // minimum non-zero value of the CDF
		for(int j=0; j<256; j++){
		    if(pdf[j] != 0) { //First non zero value will be same as pdf
				cdf_min = pdf[j]; 
				break;
			}
		}
        float val = (255.0 * float(cdf_i - cdf_min)/float(PIXELS_PER_TILE - cdf_min));
		mappings[global_i] = (unsigned char)round(val);
    }
}

void get_mappings( int* d_pdf, unsigned char* d_mappings, int num_tiles){
    
    int num_threads = 256;  
    int num_blocks = num_tiles;

    printf("Step 2 (Get mappings): Grid : {%d} blocks. Blocks : {%d} threads.\n",
    num_blocks, num_threads);
    
    ahe_get_mappings<<< num_blocks, num_threads>>>(d_pdf, d_mappings);
}

__global__ void ahe_equalize(unsigned char* img_in, unsigned char* img_out, unsigned char* mappings){
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
            unsigned char img_in_val = img_in[x+y*width];
			v00 = mappings[ img_in_val + offset00];
			v01 = mappings[ img_in_val + offset01];
			v10 = mappings[ img_in_val + offset10];
			v11 = mappings[ img_in_val + offset11];
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

__global__ void ahe_equalize_const_map(unsigned char* img_in, unsigned char* img_out){
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
            unsigned char img_in_val = img_in[x+y*width];
			v00 = Mappings[ img_in_val + offset00];
			v01 = Mappings[ img_in_val + offset01];
			v10 = Mappings[ img_in_val + offset10];
			v11 = Mappings[ img_in_val + offset11];
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
    
    int num_threads_x = 32;
    int num_threads_y = 32;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);  

    int num_blocks_x = (width_ / num_threads_x) + 1; 
    int num_blocks_y = (height_ / num_threads_y) + 1;

    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1); 

    printf("Step 3 : Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    grid_shape.x, grid_shape.y, grid_shape.z, block_shape.x, block_shape.y, block_shape.z);
    
    ahe_equalize<<< grid_shape, block_shape>>>(d_img_in, d_img_out, d_mappings);
}

void adaptive_equalization_const_mapping( unsigned char* d_img_in, unsigned char* d_img_out, unsigned char* d_mappings, int width_, int height_ , int mapping_size_bytes ){
    
    int num_threads_x = 32;
    int num_threads_y = 32;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);  

    int num_blocks_x = (width_ / num_threads_x) + 1; 
    int num_blocks_y = (height_ / num_threads_y) + 1;

    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1); 

    printf("Step 3 : Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    grid_shape.x, grid_shape.y, grid_shape.z, block_shape.x, block_shape.y, block_shape.z);

    SAFE_CALL (cudaMemcpyToSymbol, Mappings, d_mappings, mapping_size_bytes, 0, cudaMemcpyDeviceToDevice)
    ahe_equalize_const_map<<< grid_shape, block_shape>>>(d_img_in, d_img_out);
}


extern "C" void run_ahe_GPU(unsigned char* img_in, unsigned char* img_out, int width, int height){
    
	int ntiles_x = width / TILE_SIZE_X;
	int ntiles_y = height / TILE_SIZE_Y;
    int num_tiles = ntiles_x*ntiles_y;

    SAFE_CALL( cudaMemcpyToSymbol, Width, &width, sizeof(width))
    SAFE_CALL( cudaMemcpyToSymbol, Heigth, &height, sizeof(height))
    SAFE_CALL( cudaMemcpyToSymbol, Ntiles_x, &ntiles_x, sizeof(ntiles_x))
    SAFE_CALL( cudaMemcpyToSymbol, Ntiles_y, &ntiles_y, sizeof(ntiles_y))
    SAFE_CALL( cudaMemcpyToSymbol, Numtiles, &num_tiles, sizeof(num_tiles))

    int img_size = height*width;
    

    int img_size_bytes = img_size*sizeof(unsigned char);
    unsigned char *d_img_in;
    SAFE_CALL( cudaMalloc, (void**)&d_img_in, img_size_bytes)
    SAFE_CALL( cudaMemcpy, (void*)d_img_in, (void*)img_in, img_size_bytes, cudaMemcpyHostToDevice)
    
    int pdf_size = num_tiles*256;
    int pdf_size_bytes = pdf_size*sizeof(int);
    int* d_pdf;
    SAFE_CALL( cudaMalloc, (void**)&d_pdf, pdf_size_bytes)
    SAFE_CALL( cudaMemset, (void*)d_pdf, 0, pdf_size_bytes )
    get_pdf(d_img_in, d_pdf, width, height);
    SAFE_CALL( cudaDeviceSynchronize)
    
    unsigned char* d_mappings;
    int mappings_size = num_tiles*256;
    int mappings_size_bytes = pdf_size*sizeof(unsigned char);
    SAFE_CALL( cudaMalloc, (void**)&d_mappings, mappings_size_bytes)
    get_mappings(d_pdf, d_mappings, num_tiles);
    SAFE_CALL( cudaDeviceSynchronize )

    print_mappings(d_mappings, mappings_size, "mappings_GPU");
    
    unsigned char *d_img_out;
    SAFE_CALL( cudaMalloc, (void**)&d_img_out, img_size_bytes)
    // adaptive_equalization(d_img_in, d_img_out, d_mappings, width, height);
    adaptive_equalization_const_mapping(d_img_in, d_img_out, d_mappings, width, height, mappings_size_bytes);
    SAFE_CALL(cudaDeviceSynchronize)

    SAFE_CALL(cudaMemcpy, (void*)img_out, (void*)d_img_out,img_size_bytes, cudaMemcpyDeviceToHost)

    SAFE_CALL( cudaFree, d_img_in)
    SAFE_CALL( cudaFree, d_mappings)
    SAFE_CALL (cudaFree, d_img_out)
}