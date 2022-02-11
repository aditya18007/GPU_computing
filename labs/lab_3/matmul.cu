#include <iostream>

#include "utils.h"
#include "defines.h"

#define M DIM


__global__ 
void matmul_basic(int* A, int* B, int* C){
    int x = threadIdx.x  + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    if ( x < M && y < M){
        int result = 0;
        for( int k = 0; k < M; k++){
            result += A[ x*M + k]*B[k*M + y];
        }
        C[ x*M + y] = result;
    }
}

void run_matmul_basic( GPU_array<int>& d_A, GPU_array<int>& d_B, GPU_array<int>& d_C){
    dim3 block_dim(TILE, TILE, 1);
    dim3 grid_dim( (M/TILE)+1, (M/TILE)+1, 1);
    matmul_basic<<<grid_dim, block_dim>>>(d_A.arr(), d_B.arr(), d_C.arr());
}

using namespace std; 
int main(){
    cudaEvent_t start, end;
    SAFE_CALL( cudaEventCreate, &start)
    SAFE_CALL( cudaEventCreate, &end)

    CPU_array<int> A(M*M), B(M*M);
    for( int i = 1; i <= M*M; i++){
        A(i-1) = i;
        B(i-1) = i;
    }
    GPU_array<int> d_A(A), d_B(B), d_C(M*M);
    
    SAFE_CALL(cudaEventRecord, start)
    run_matmul_basic(d_A, d_B, d_C);
    SAFE_CALL( cudaDeviceSynchronize )
    SAFE_CALL( cudaEventRecord, end )
    SAFE_CALL (cudaEventSynchronize, end )
    float time;
    SAFE_CALL (cudaEventElapsedTime, &time, start, end);
    cout << "~~~~~~~~~~~~~~GPU EXECUTION~~~~~~~~~~~~~~\n";
    cout << "Dimension size = " << M << '\n';
    cout << "Time taken = " << time << "ms" << endl;
    write_to_file(d_C, "GPU_basic");
}