#include <iostream>

#include "utils.h"
#include "defines.h"

#define M DIM


__global__ 
void matmul_shared(int* global_A, int* global_B, int* global_C){
    __shared__ int A[TILE][TILE];
    __shared__ int B[TILE][TILE];

    int row = threadIdx.y  + blockDim.y*blockIdx.y;
    int col = threadIdx.x  + blockDim.x*blockIdx.x;
    
    int result = 0;
    for( int i = 0; i < gridDim.x ; i++){
        int blk_row = i*TILE + threadIdx.y;
        int blk_col = i*TILE + threadIdx.x;
        if( blk_col < M && row < M){
            A[threadIdx.y][threadIdx.x] = global_A[row*M + blk_col];
        } else {
            A[threadIdx.y][threadIdx.x] = 0;
        }
                
        if ( col < M && blk_row < M ) {
            B[threadIdx.y][threadIdx.x] = global_B[blk_row*M + col];
        } else {
            B[threadIdx.y][threadIdx.x] = 0;
        }
                
        __syncthreads();
        for( int k = 0; k < TILE; k++){
            result += A[threadIdx.y][k]*B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if( col < M && row < M){
        global_C[ row*M + col] = result;  
    }
    
}

void run_matmul_shared( GPU_array<int>& d_A, GPU_array<int>& d_B, GPU_array<int>& d_C){
    dim3 block_dim(TILE, TILE, 1);
    dim3 grid_dim( (M/TILE)+1, (M/TILE)+1, 1);
    matmul_shared<<<grid_dim, block_dim>>>(d_A.arr(), d_B.arr(), d_C.arr());
}

__global__ 
void matmul_basic(int* A, int* B, int* C){
    int row = threadIdx.y  + blockDim.y*blockIdx.y;
    int col = threadIdx.x  + blockDim.x*blockIdx.x;
    if ( row < M && col < M){
        int result = 0;
        for( int k = 0; k < M; k++){
            result += A[ row*M + k]*B[k*M + col];
        }
        C[ row*M + col] = result;
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
    run_matmul_shared(d_A, d_B, d_C);
    SAFE_CALL( cudaDeviceSynchronize )
    SAFE_CALL( cudaEventRecord, end )
    SAFE_CALL (cudaEventSynchronize, end )
    float time;
    SAFE_CALL (cudaEventElapsedTime, &time, start, end);
    cout << "~~~~~~~~~~~~~~GPU EXECUTION~~~~~~~~~~~~~~\n";
    cout << "Dimension size = " << M << '\n';
    cout << "Grid Dim = " << ((M/TILE)+1) << '\n';
    cout << "Time taken = " << time << "ms" << endl;
    write_to_file(d_C, "GPU");
}