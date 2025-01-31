#include <iostream>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime.h>
#include "utils.h"

#define ARRAY_SIZE 512

using namespace std;


struct record
{
    int key, value;
};


__global__ void vector_add(struct record* a, struct record* b, struct record* c){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i >= ARRAY_SIZE){
        return;
    }
    c[i].key = a[i].key + b[i].key;
    c[i].value = a[i].value + b[i].value;
}


int main(){

    CPU_array<struct record> AoS_data1(ARRAY_SIZE), AoS_data2(ARRAY_SIZE);
    
    for(int i = 0; i < ARRAY_SIZE; i++){
        AoS_data1(i).key = i;
        AoS_data1(i).value = i;

        AoS_data2(i).key = i;
        AoS_data2(i).value = i;
    }

    // cudaMalloc
    GPU_array<struct record> d_AoS_data1(AoS_data1), d_AoS_data2(AoS_data2), d_AoS_data3(ARRAY_SIZE);

    vector_add<<<(ARRAY_SIZE/256)+1, 256>>>(d_AoS_data1.arr(), d_AoS_data2.arr(), d_AoS_data3.arr());

    cudaDeviceSynchronize();
    CPU_array<struct record> AoS_data3(d_AoS_data3);
}