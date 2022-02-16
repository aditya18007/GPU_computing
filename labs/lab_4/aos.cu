#include <iostream>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime.h>

#define ARRAY_SIZE 512

using namespace std;


struct record
{
    int a, b, c;
};


__global__ void vector_add(record* a, record* b, record* c){
    int i = threadIdx.x ;
	........
}


int main(){
    struct record *AoS_data1, *AoS_data2, *AoS_data3 , *d_AoS_data1, *d_AoS_data2, *d_AoS_data3;
    
    
    // malloc AoSdata1, AoS_data2, AoSdata3
    ......
    ......
    ......

    // initialize array keys, values
    for (int i = 0; i < ARRAY_SIZE; i ++){
        .....
    }

    // cudaMalloc
    .....
    .....
    .....

    // cudaMemcpy
    .....
    .....
    .....


    vector_add<<<..............>>>(d_AoS_data1, d_AoS_data2, d_AoS_data3);

    cudaDeviceSynchronize();
    
    // cudaMemcpy back to host array
    ......   
}