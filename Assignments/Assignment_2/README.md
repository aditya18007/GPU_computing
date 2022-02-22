# Assignment-2

| Roll No |         Name         |
| :-----: | :------------------: |
| 2018007 | Aditya Singh Rathore |



## Part 1 (SDT computation on GPU )

> (a) Write a CUDA version of the SDT computation using shared memory.
> (b) Document your approach to the problem. 
> (c) Perform computations on CPU and GPU with image of sizes (256, 512, 1024, and 
> 2048).  Tabulate  CPU  and  GPU  (kernel  and  overall)  timing  results,  plot  speedups 
> (kernel and overall),  and report the`MSE` error in each case.

**Note** : Most optimised approach explained here.

* The idea is compute minimum distances of each pixel from edge in one kernel and compute SDT from those minimum distances in second kernel.

* The number of edge pixels is calculated on the GPU.

  * Rather than atomic add to global variable for each thread, we have a block level variable and all threads add count atomically to that variable. 
  * Once it has counted edge pixels in the given block, we can add the block level count to global count (once per block). 

* Edge pixels are computed on the CPU. We cannot have multiple pixels elements writing to edge array, using atomic ops its a serial operation.

* Once we calculate minimum distance for each pixel, SDT calculation is straight forward.

* For computing minimum distance, we have to do the following:

  * ```python
    for pixel in image:
    	d_min = inf
    	a, b = pixel coordinates
        for edge in edges:
    		x, y  = edge coordinates
            d = (a-x)^2 + (b-y)^2
            d_min = min(d_min, d)
        min_dist[pixel] = d_min
    ```

* If we see calculating d, we can expand it further as:

  * ```
    d = (a-x)^2 + (b-y)^2
    d = a^2 + b^2 + x^2 + y^2 -2*(ax + by)
    ```

* We are computing a lot of those terms again and again. We can pre-compute them once and use them again as follows

  * ```python
    struct edge{
    	x, y, sqr
    }
    Edges = struct edge[number of edges]
    for edge in edges:
        x, y = edge coordinates
        Edges[edge].x = x
        Edges[edge].y = y
        Edges[edge].sqr = x^2 + y^2
    
    for pixel in image:
    	d_min = inf
    	a, b = pixel coordinates
        sqr = a^2 + b^2
        for edge in Edges:
            d = sqr + edge.sqr - 2*(a*edge.x + b*edge.y)
            d_min = min(d_min, d)
        min_dist[pixel] = d_min
    ```

* We can see that each pixel will access each edge. We can store edges in shared memory for faster access. 

* But, size of edges can be very large. So we will divide the edges into chunks of 1024 and call the kernel again and again with each chunk to get the global minimum distance. (see optimization why 1024)

  * ```python
    Edges = | c1 | c2 | c3 | ...| cn-1 |cn |
    sz_edge = len(Edges)
    size of c1 ... cn-1 = 1024
    size of cn = sz_edge % 1024
    ```

  * Calculating minimum distance for each pixel using chunks

    ```python
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
    ```





## Part 2

> How  will  you  modify  your  approach  to  use  constant  memory  instead  of  shared 
> memory? Explain why using constant memory instead of shared memory is a good/bad 
> choice in this case.

**Note** : Done for the most optimised approach.



## Part 3 (Kernel Analysis)

> (a) Analyze your CUDA kernel in terms of efficiency using `nvprof`/`nvvp` tool. 
> (b) Identify bottlenecks in your kernel.



## Part 4 (Kernel Optimization)



## Appendix

### Kernel-1

* Compute Edges on the CPU.

* All Computations in one kernel. 
* Each thread will compute SDT for each pixel.

```c++
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
```

### Kernel-2

* Compute minimum distance for each pixel in different kernel.
* Compute `sdt` for each pixel in a different Kernel .
* For computing minimum distance, we will store the edges in shared memory because each pixel reads all edges, each block can store in shared memory which will be faster.
* We cannot store entire edge array in shared memory as it gets very large, so we will divide it into chunks, say 25 and call the kernel multiple times.  
* For each pixel, we will initialise minimum distance to `FLT_MAX`.
* We will find the minimum distance with each chunk, first with chunk-1. Then with chunk-2 and so on. This way we will have the global minimum distance. 

#### Kernel

```c++
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
```

#### Calling 

```c++
const auto num_chunks = 25;
const auto chunk_size = sz_edge/num_chunks;
const auto last_chunk = sz_edge%num_chunks;
const auto block_size = 256;	
const auto grid_size = (sz/block_size) + 1; 
for(int i = 0; i < num_chunks; i++){
	compute_dist<<< grid_size, block_size, chunk_size*sizeof(int)>>>(d_min_dist.arr(), 
	d_edges.arr(), 
	height, width,
    i*chunk_size, 
    chunk_size);
}
if (last_chunk != 0){
	compute_dist<<< grid_size, block_size, last_chunk*sizeof(int)>>>(d_min_dist.arr(),
		d_edges.arr(),
		height, width,
		num_chunks*chunk_size, last_chunk);
}
```

### Kernel-3

* Same as Kernel-2. 
* Global load store efficiency is improved. 
* Edges is divided into equal sized chunks of size 1024.
* Instead of first thread in block writing to shared memory, all threads write to shared memory (chunk size = block size).

#### Kernel

```c++
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
```

#### Launch

````c++
const auto chunk_size = 1024;
const auto num_chunks = sz_edge/chunk_size;
const auto last_chunk = sz_edge%chunk_size;
	
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
````

### Kernel-4

* When computing `dist2`, each thread will be computing some computations that can be done only once at block level (explained in Kernel Optimization).

#### Kernel

```c++
__global__ void compute_dist(float* min_dist, int* global_edges,int start)
{
	extern __shared__ int edges[];
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int edge = global_edges[threadIdx.x+start];
	int _x = edge % Width[0];
	int _y = edge / Width[0];
	edges[3*threadIdx.x] = _x;
	edges[3*threadIdx.x+1] = _y;
	edges[3*threadIdx.x+2] = _x*_x + _y*_y;
	__syncthreads();

	int sz = Sz[0];
	if (i >= sz) return;
	
	int x = i%Width[0];
	int y = i/Width[0];
	float sqr = x*x + y*y;
	float min = min_dist[i];
	float dist2;
	for(int k = 0; k < blockDim.x; k++){
		dist2 =  edges[3*k+2] + sqr -2*(x*edges[3*k] + y*edges[3*k+1]);
		if(dist2 < min) min = dist2;
	}

	min_dist[i] = min;
}
```



### Kernel-5

* We can count number of edge pixels on the GPU.
* We are calculating `x`, `y` and `sqr` for each edge in every block. 
* We can still do better by computing them globally and creating an `array of structs` for each each edge.

#### Kernel

```c++
struct Edge{
	float x, y, sqr;
};
```

* Count the number of pixels.

```c++
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
```

* Calculate x, y and sqr for each edge index calculated on the CPU.

```c++
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
```

* We create shared memory of `struct Edge`

```c++
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
```

