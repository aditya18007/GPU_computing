#include <iostream>
#include <stdio.h>
#include <time.h>
#include "utils.h"
#define LENGTH 1000

using namespace std;

// Define constant memory arrays A and B of size LENGTH
__constant__ float A[LENGTH];
__constant__ float B[LENGTH];
// Write kernel for initiailizing as C[i] = A[blockIdx.x] + B[blockIdx.x]

__global__ void initialize2(float *C){
    // Code here
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < LENGTH){
        C[i] = A[blockIdx.x] + B[blockIdx.x];
    }
}


int main(){
    float *a, *b;
    float *d_c;

    a = new float[LENGTH];
    b = new float[LENGTH];

    // Allocate space on the device for d_c
    SAFE_CALL(cudaMalloc, (void**)&d_c , LENGTH * sizeof(float))
    for (int i = 0; i < LENGTH; i ++){
        a[i] = i;
        b[i] = i;
    }

    // Transfering data from host variables a and b to constant variables A and B, respectively
    cudaMemcpyToSymbol(A, a, LENGTH*sizeof(float));
    cudaMemcpyToSymbol(B, b, LENGTH*sizeof(float));


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    float milliseconds = 0;


    float time = 0;
    const int block_size = 128;
    const int num_blocks = 1 + (LENGTH/block_size);
    
    for (int i = 0; i < 1000; i ++){
        cudaEventSynchronize(start);
        cudaEventRecord(start);

       // Create kernel with block size = 128 threads. Call the initialize2 kernel. 
        initialize2<<<num_blocks,block_size>>>(d_c);
        // (Memcopy from device to host not required for this lab)
        cudaDeviceSynchronize();
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&milliseconds, start, end);
        time += milliseconds;
    }

    std::cout<< "Initialize 2 : "<< time/1000 << std::endl; 

    // Free the memory of device and host variables
    SAFE_CALL( cudaFree, d_c)

    delete []a;
    delete []b; 

    return 0;
}