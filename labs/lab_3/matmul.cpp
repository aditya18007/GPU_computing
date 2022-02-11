#include <chrono>
#include <vector>
#include <iostream>


constexpr size_t M = 100;

using namespace std;
using namespace std::chrono;

void matmul(vector<int>& A, vector<int>& B, vector<int>& C ){
    for( int row = 0; row < M; row++ ){
        cout << row << "\n";
        for( int col = 0; col < M; col++){
            int result = 0;
            for( int k = 0; k < M; k++){
                result += A[ row*M + k]*B[k*M + col];
            }   
            C[ row*M + col] = result;
        }
    }
}

int main(){
    vector<int> A(M*M);
    vector<int> B(M*M);
    vector<int> C(M*M);

    for( int i = 1; i <= M*M; i++){
        A[i-1] = i;
        B[i-1] = i;
    }

    auto start = high_resolution_clock::now();
    matmul(A, B, C);
    auto end = high_resolution_clock::now();
 
    auto duration = duration_cast<nanoseconds>(end - start);
    auto time = duration.count() / 1e6;
 
    cout << "~~~~~~~~~~~~~~CPU EXECUTION~~~~~~~~~~~~~~\n";
    cout << "Dimension size = " << M << '\n';
    cout << "Time taken = " << time << "ms" << endl;
}