#include "utils.h"
#include <float.h>
#include <fstream>

cudaError_t my_errno;

__global__ void compute_sdt(float* sdt, unsigned char* bitmap, int* edges, int height, int width, int edge_size)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if ( x < width && y < height){
		float min_dist = FLT_MAX;
		for(int k=0; k<edge_size; k++)
		{
			float _x = edges[k] % width;
			float _y = edges[k] / width;
			float dx = _x - x;
			float dy = _y - y;
			float dist2 = dx*dx + dy*dy;
			if(dist2 < min_dist) min_dist = dist2;
      }
      float sign  = (bitmap[x + y*width] >= 127)? 1.0f : -1.0f;
      sdt[x + y*width] = sign * sqrtf(min_dist);
	}
}

extern "C" void gpu_main(unsigned char* bitmap, float *sdt, int width, int height)
{
	size_t sz = width*height;
    

	int sz_edge = 0;
  	for(int i = 0; i<sz; i++) if(bitmap[i] == 255) sz_edge++;
  	int *edge_pixels = new int[sz_edge];
  	for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;
	
	GPU_array<int> d_edges(edge_pixels, sz_edge);
	GPU_array<float> d_sdt(sz);
	GPU_array<unsigned char> d_bitmap(bitmap, sz);
	dim3 block_dim(32, 32, 1);
	dim3 grid_dim((width/16)+1, (height/16) + 1, 1);
	compute_sdt<<< grid_dim, block_dim>>>(d_sdt.arr(), d_bitmap.arr(), d_edges.arr(), height, width, sz_edge);
	cudaDeviceSynchronize();
	d_sdt.write_to_ptr(sdt);
}