#include "utils.h"
#include <float.h>
#include <fstream>

cudaError_t my_errno;

__global__ void compute_dist(float* min_dist, int* global_edges, int height, int width, int start, int chunk_size)
{
	extern __shared__ int edges[];
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(threadIdx.x == 0){
		//Leader thread will write chunk to memory
		for(int j = 0; j < chunk_size; j++){
			edges[j] = global_edges[j+start]; 
		}
	}
	__syncthreads();
	if (i >= height*width) return;

	int x = i%width;
	int y = i/width;
	float min = min_dist[i];
	float _x, _y, dx, dy, dist2;
	for(int k = 0; k < chunk_size; k++){
		_x = edges[k] % width;
		_y = edges[k] / width;
		dx = _x - x;
		dy = _y - y;
		dist2 = dx*dx + dy*dy;
		if(dist2 < min) min = dist2;
	}
	min_dist[i] = min;
}

__global__ void compute_sdt(unsigned char* bitmap, float* min_dist, float* sdt, int sz){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i >= sz) return;
	float sign  = (bitmap[i] >= 127)? 1.0f : -1.0f;
    sdt[i] = sign * sqrtf(min_dist[i]);
}

extern "C" void gpu_main(unsigned char* bitmap, float *sdt, int width, int height)
{
	size_t sz = width*height;
    

	int sz_edge = 0;
  	for(int i = 0; i<sz; i++) if(bitmap[i] == 255) sz_edge++;
  	int *edge_pixels = new int[sz_edge];
  	for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;
	
	GPU_array<int> d_edges(edge_pixels, sz_edge);
	CPU_array<float> min_dist(sz);
	for(int i = 0; i < sz; i++){
		min_dist(i) = FLT_MAX;
	}
	GPU_array<float> d_min_dist(min_dist);
	
	const auto num_chunks = 25;
	const auto chunk_size = sz_edge/num_chunks;
	const auto last_chunk = sz_edge%num_chunks;
	const auto block_size = 256;
	const auto grid_size = (sz/block_size) + 1; 
	for(int i = 0; i < num_chunks; i++){
		compute_dist<<< grid_size, block_size, chunk_size*sizeof(int)>>>(d_min_dist.arr(), 
		d_edges.arr(), 
		height, width, 
		i*chunk_size, chunk_size);
	}
	if (last_chunk != 0){
		compute_dist<<< grid_size, block_size, last_chunk*sizeof(int)>>>(d_min_dist.arr(),
			d_edges.arr(),
			height, width,
			num_chunks*chunk_size, last_chunk);
	}
		
	GPU_array<unsigned char> d_bitmap(bitmap, sz);
	GPU_array<float> d_sdt(sz);
	compute_sdt<<< grid_size, block_size >>>(d_bitmap.arr(), d_min_dist.arr(), d_sdt.arr(), sz);
	cudaDeviceSynchronize();
	d_sdt.write_to_ptr(sdt);
}