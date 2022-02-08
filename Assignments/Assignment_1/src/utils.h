#ifndef MY_GPU_UTILS_H
#define MY_GPU_UTILS_H
#include <cuda_runtime.h>

cudaError_t my_errno ;
#define SAFE_CALL( f_call, ... ) \
    my_errno = f_call(__VA_ARGS__); \
    if ( my_errno != cudaSuccess ){ \
        std::cerr << "\nFile : " << __FILE__ << '\n'; \
        std::cerr << "Function : " << __func__ << '\n'; \
        std::cerr << "Line : " << __LINE__ << '\n'; \
        std::cerr << "Cuda error : " << cudaGetErrorString(my_errno) << " !\n\n"; \
        exit(EXIT_FAILURE);\
    } \

#endif //MY_GPU_UTILS_H