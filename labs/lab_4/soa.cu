#include <iostream>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime.h>

#define ARRAY_SIZE 512

using namespace std;


struct SoA
{
    int *keys;
    int *values;
};

__global__ void vector_add(SoA a, SoA b, SoA c){
    int i = threadIdx.x ;
	.......
}


int main(){
    struct SoA SoA_data1, SoA_data2, SoA_data3 ,d_SoA_data1, d_SoA_data2, d_SoA_data3;
    
    // malloc SoAdata1.keys, SoAdata1.values SoA_data2.keys, SoAdata3.keys etc
    ......
    ......
    ......

    // initialize array keys, values
    for (int i = 0; i < ARRAY_SIZE; i ++){
        ......
    }


    // cudaMalloc d_SoA_data1.keys, d_SoA_data1.values etc
    .....
    .....
    .....
    .....
    .....
    .....


    // cudaMemcpy d_SoA_data1.keys, d_SoA_data1.values etc
    .....

    vector_add<<<................>>>(d_SoA_data1, d_SoA_data2, d_SoA_data3);

    cudaDeviceSynchronize();
    
    // copy back to host array
    ......
       
}