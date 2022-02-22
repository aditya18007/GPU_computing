#include "utils.h"
#include <float.h>
#include <fstream>

cudaError_t my_errno;

__constant__ int Width[1];
__constant__ int Sz[1];
__constant__ int Sz_edges[1];

struct Edge{
	float x, y, sqr;
};

__global__ void count_edge_pixels(unsigned char* bitmap, int* sz_edges){
	__shared__ int num_edges;
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i >= Sz[0]) return;

	if (threadIdx.x == 0){
		num_edges = 0;
	}
	__syncthreads();
	
	if (bitmap[i] == 255){
		atomicAdd(&num_edges, 1);
	}
	__syncthreads();
	if (threadIdx.x == 0){
		atomicAdd(sz_edges, num_edges);
	}
}

__global__ void compute_edge_pixels(struct Edge* edges, int* edge_indices){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i >= Sz_edges[0]) return;
	int edge = edge_indices[i];
	int _x = edge % Width[0];
	int _y = edge / Width[0];
	edges[i].x = _x;
	edges[i].y = _y;
	edges[i].sqr = _x*_x + _y*_y;
}

__global__ void compute_dist(float* min_dist, struct Edge* global_edges,int start)
{
	extern __shared__ struct Edge edges[];
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	edges[threadIdx.x] = global_edges[start + threadIdx.x];	
	__syncthreads();
	if (i >= Sz[0]) return;
	int x = i%Width[0];
	int y = i/Width[0];
	float sqr = x*x + y*y;
	float min = min_dist[i];
	float dist2;
	for(int k = 0; k < blockDim.x; k++){
		dist2 =  edges[k].sqr + sqr -2*(x*edges[k].x + y*edges[k].y);
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
	int sz = width*height;
    SAFE_CALL( cudaMemcpyToSymbol, Width, &width, sizeof(width))
	SAFE_CALL( cudaMemcpyToSymbol, Sz, &sz, sizeof(sz))
	GPU_array<unsigned char> d_bitmap(bitmap, sz);
	
	//Count edge pixels
	CPU_array<int> sz_edges(1);
	sz_edges(0) = 0;
	GPU_array<int> d_sz_edges(sz_edges);
	count_edge_pixels<<<(sz/1024) + 1, 1024>>>(d_bitmap.arr(), d_sz_edges.arr());
	int sz_edge = 0;

	d_sz_edges.write_to_ptr(&sz_edge);
	SAFE_CALL( cudaMemcpyToSymbol, Sz_edges, &sz_edge, sizeof(sz_edge))
  	
	//Record edge pixels
	int *edge_pixels = new int[sz_edge];
  	for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;
	
	//Calculate the x, y and sqr
	GPU_array<int> d_edge_indices(edge_pixels, sz_edge);
	GPU_array<struct Edge> d_edges(sz_edge);
	compute_edge_pixels<<< (sz_edge/1024)+1, 1024>>>(d_edges.arr(), d_edge_indices.arr());

	//Compute minimum distance
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
		compute_dist<<< grid_size, block_size, chunk_size*sizeof(struct Edge)>>>(d_min_dist.arr(), 
		d_edges.arr(), 
		i*chunk_size);
	}
	if (last_chunk != 0){
		compute_dist<<< grid_last_chunk, last_chunk, last_chunk*sizeof(struct Edge)>>>(d_min_dist.arr(),
			d_edges.arr(),
			num_chunks*chunk_size);
	}
		
	//Compute SDT
	GPU_array<float> d_sdt(sz);
	compute_sdt<<< grid_size, block_size >>>(d_bitmap.arr(), d_min_dist.arr(), d_sdt.arr());
	
	SAFE_CALL(cudaDeviceSynchronize)
	d_sdt.write_to_ptr(sdt);
}