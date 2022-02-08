#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "ahe_gpu_kernels.fatbin.c"
extern void __device_stub__Z13sample_kernelv(void);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z13sample_kernelv(void){__cudaLaunchPrologue(1);__cudaLaunch(((char *)((void ( *)(void))sample_kernel)));}
# 5 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
void sample_kernel(void)
# 6 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
{__device_stub__Z13sample_kernelv();


}
# 1 "ahe_gpu_kernels.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T0) {  __nv_dummy_param_ref(__T0); __nv_save_fatbinhandle_for_managed_rt(__T0); __cudaRegisterEntry(__T0, ((void ( *)(void))sample_kernel), _Z13sample_kernelv, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
