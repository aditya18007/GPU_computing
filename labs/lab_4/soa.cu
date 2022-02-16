#include <iostream>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime.h>
#include "utils.h"

#define ARRAY_SIZE 512

using namespace std;


struct SoA
{
    int *keys;
    int *values;
};

__global__ void vector_add(SoA a, SoA b, SoA c){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
	if ( i >= ARRAY_SIZE){
        return;
    }
    c.keys[i] = a.keys[i] + b.keys[i];
    c.values[i] = a.values[i] + b.values[i];
}


int main(){
    struct SoA SoA_data1, SoA_data2, d_SoA_data1, d_SoA_data2, d_SoA_data3;
    
    // malloc SoAdata1.keys, SoAdata1.values SoA_data2.keys, SoAdata3.keys etc
    SoA_data1.keys = new int[ARRAY_SIZE];
    SoA_data1.values = new int[ARRAY_SIZE];

    SoA_data2.keys = new int[ARRAY_SIZE];
    SoA_data2.values = new int[ARRAY_SIZE];

    // initialize array keys, values
    for (int i = 0; i < ARRAY_SIZE; i ++){
        SoA_data1.keys[i] = i;
        SoA_data1.values[i] = i;

        SoA_data2.keys[i] = i;
        SoA_data2.values[i] = i;
    }


    // cudaMalloc d_SoA_data1.keys, d_SoA_data1.values etc
    SAFE_CALL( cudaMalloc, (void**)&d_SoA_data1.keys, sizeof(int)*ARRAY_SIZE)
    SAFE_CALL( cudaMalloc, (void**)&d_SoA_data1.values, sizeof(int)*ARRAY_SIZE)

    SAFE_CALL( cudaMalloc, (void**)&d_SoA_data2.keys, sizeof(int)*ARRAY_SIZE)
    SAFE_CALL( cudaMalloc, (void**)&d_SoA_data2.values, sizeof(int)*ARRAY_SIZE)

    SAFE_CALL( cudaMalloc, (void**)&d_SoA_data3.keys, sizeof(int)*ARRAY_SIZE)
    SAFE_CALL( cudaMalloc, (void**)&d_SoA_data3.values, sizeof(int)*ARRAY_SIZE)

    // cudaMemcpy d_SoA_data1.keys, d_SoA_data1.values etc
    SAFE_CALL( cudaMemcpy, d_SoA_data1.keys, SoA_data1.keys, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice)
    SAFE_CALL( cudaMemcpy, d_SoA_data1.values, SoA_data1.values, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice)

    SAFE_CALL( cudaMemcpy, d_SoA_data2.keys, SoA_data2.keys, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice)
    SAFE_CALL( cudaMemcpy, d_SoA_data2.values, SoA_data2.values, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice)

    vector_add<<< (ARRAY_SIZE/256)+1,256 >>>(d_SoA_data1, d_SoA_data2, d_SoA_data3);

    cudaDeviceSynchronize();

    
    SAFE_CALL( cudaFree, d_SoA_data1.keys)
    SAFE_CALL( cudaFree, d_SoA_data1.values)

    SAFE_CALL( cudaFree, d_SoA_data2.keys)
    SAFE_CALL( cudaFree, d_SoA_data2.values)

    SAFE_CALL( cudaFree, d_SoA_data3.keys)
    SAFE_CALL( cudaFree, d_SoA_data3.values)

    delete[] SoA_data1.keys;
    delete[] SoA_data1.values;
    delete[] SoA_data2.keys;
    delete[] SoA_data2.values;

}