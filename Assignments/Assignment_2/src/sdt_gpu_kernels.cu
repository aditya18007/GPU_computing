#include "utils.h"
#include <float.h>
#include <fstream>

cudaError_t my_errno;
#define EDGES 2881

struct edge{
	int x,y;
	float sqr;
};

__constant__ int Width[1];
__constant__ int Height[1];
__constant__ int Sz[1];
__constant__ int Sz_edge[1];
__constant__ struct edge Edges[EDGES];

__global__ void compute_dist_const(float* min_dist, int start){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i >= Sz[0]) return;
	int x = i%Width[0];
	int y = i/Width[0];
	float sqr = x*x + y*y;
	float min = min_dist[i];
	float dist2;
	for(int k = 0; k < blockDim.x; k++){
		dist2 =  Edges[k+start].sqr + sqr -2*(x*Edges[k+start].x + y*Edges[k+start].y);
		if(dist2 < min) min = dist2;
	}
	min_dist[i] = min;
}

__global__ void compute_sdt(unsigned char* bitmap, float* min_dist, float* sdt){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i >= Sz[0]) return;
	float sign  = (bitmap[i] >= 127)? 1.0f : -1.0f;
    sdt[i] = sign * sqrt(min_dist[i]);
}

extern "C" void gpu_main(unsigned char* bitmap, float *sdt, int width, int height)
{
	size_t sz = width*height;
    

	int sz_edge = 0;
  	for(int i = 0; i<sz; i++) if(bitmap[i] == 255) sz_edge++;
	if(EDGES == sz_edge){
		printf("Bingo\n");
	}
	struct edge *edge_pixels = new struct edge[sz_edge];
  	for(int i = 0, j = 0; i<sz; i++) {
		if(bitmap[i] == 255) {
			int x = i % width;
			int y = i / width;
			edge_pixels[j].x = x;
			edge_pixels[j].y = y;
			edge_pixels[j].sqr = x*x + y*y;
			j++;
		}
	}
	
	SAFE_CALL( cudaMemcpyToSymbol, Width, &width, sizeof(width))
	SAFE_CALL( cudaMemcpyToSymbol, Height, &height, sizeof(height))
	SAFE_CALL( cudaMemcpyToSymbol, Sz, &sz, sizeof(sz))
	SAFE_CALL( cudaMemcpyToSymbol, Sz_edge, &sz, sizeof(sz_edge))
	SAFE_CALL( cudaMemcpyToSymbol, Edges, edge_pixels, EDGES*sizeof(struct edge))

	GPU_array<struct edge> d_edges(edge_pixels, sz_edge);
	CPU_array<float> min_dist(sz);
	for(int i = 0; i < sz; i++){
		min_dist(i) = FLT_MAX;
	}
	GPU_array<float> d_min_dist(min_dist);
	
	const auto chunk_size = 1024;
	const auto num_chunks = sz_edge/chunk_size;
	const auto last_chunk = sz_edge%chunk_size;
	
	const auto block_size = chunk_size;
	const auto grid_size = (sz/block_size) + 1;
	const auto grid_last_chunk = (sz/last_chunk) + 1;
	for(int i = 0; i < num_chunks; i++){
		compute_dist_const<<< grid_size, block_size>>>(d_min_dist.arr(), 
		i*chunk_size);
		std::cout << i*chunk_size << std::endl;
	}
	if (last_chunk != 0){
		compute_dist_const<<< grid_last_chunk, last_chunk>>>(d_min_dist.arr(),
			num_chunks*chunk_size);
		std::cout << num_chunks*chunk_size << std::endl;
	}
	
	GPU_array<unsigned char> d_bitmap(bitmap, sz);
	GPU_array<float> d_sdt(sz);
	compute_sdt<<< grid_size, block_size >>>(d_bitmap.arr(), d_min_dist.arr(), d_sdt.arr());
	cudaDeviceSynchronize();
	d_sdt.write_to_ptr(sdt);
}