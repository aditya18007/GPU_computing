#include "utils.h"
#include <float.h>
#include <fstream>

cudaError_t my_errno;

__constant__ int Width[1];
__constant__ int Height[1];
__constant__ int Sz[1];
__constant__ int Sz_edge[1];

__global__ void init_memory(float *min_dist, int size){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i >= size) return;
	min_dist[i] = FLT_MAX;
}

__global__ void compute_dist(float* min_dist, int* global_edges,int start)
{
	extern __shared__ int edges[];
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	edges[threadIdx.x] = global_edges[threadIdx.x+start];
	__syncthreads();

	int sz = Sz[0];
	if (i >= sz) return;
	
	int width = Width[0];
	
	int x = i%width;
	int y = i/width;
	float min = min_dist[i];
	float _x, _y, dx, dy, dist2;
	for(int k = 0; k < blockDim.x; k++){
		int edge = edges[k];
		_x = edge % width;
		_y = edge / width;
		dx = _x - x;
		dy = _y - y;
		dist2 = dx*dx + dy*dy;
		if(dist2 < min) min = dist2;
	}
	min_dist[i] = min;
}

__global__ void compute_sdt(unsigned char* bitmap, float* min_dist, float* sdt){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i >= Sz[0]) return;
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
	SAFE_CALL( cudaMemcpyToSymbol, Width, &width, sizeof(width))
	SAFE_CALL( cudaMemcpyToSymbol, Height, &height, sizeof(height))
	SAFE_CALL( cudaMemcpyToSymbol, Sz, &sz, sizeof(sz))
	SAFE_CALL( cudaMemcpyToSymbol, Sz_edge, &sz, sizeof(sz_edge))

	GPU_array<int> d_edges(edge_pixels, sz_edge);
	CPU_array<float> min_dist(sz);
	for(int i = 0; i < sz; i++){
		min_dist(i) = FLT_MAX;
	}
	GPU_array<float> d_min_dist(min_dist);
	
	const auto chunk_size = 1024;
	const auto num_chunks = sz_edge/chunk_size;
	const auto last_chunk = sz_edge%chunk_size;
	printf("Num chunks = %d\nChunk Size = %d\n", num_chunks, chunk_size);
	
	const auto block_size = chunk_size;
	const auto grid_size = (sz/block_size) + 1;
	const auto grid_last_chunk = (sz/last_chunk) + 1;
	for(int i = 0; i < num_chunks; i++){
		compute_dist<<< grid_size, block_size, chunk_size*sizeof(int)>>>(d_min_dist.arr(), 
		d_edges.arr(), 
		i*chunk_size);
	}
	if (last_chunk != 0){
		compute_dist<<< grid_last_chunk, last_chunk, last_chunk*sizeof(int)>>>(d_min_dist.arr(),
			d_edges.arr(),
			num_chunks*chunk_size);
	}
		
	GPU_array<unsigned char> d_bitmap(bitmap, sz);
	GPU_array<float> d_sdt(sz);
	compute_sdt<<< grid_size, block_size >>>(d_bitmap.arr(), d_min_dist.arr(), d_sdt.arr());
	cudaDeviceSynchronize();
	d_sdt.write_to_ptr(sdt);
}