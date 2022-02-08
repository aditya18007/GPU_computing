#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "ahe_gpu_kernels.fatbin.c"
extern void __device_stub__Z7ahe_GPUPh(unsigned char *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z7ahe_GPUPh(unsigned char *__par0){__cudaLaunchPrologue(1);__cudaSetupArgSimple(__par0, 0UL);__cudaLaunch(((char *)((void ( *)(unsigned char *))ahe_GPU)));}
# 14 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
void ahe_GPU( unsigned char *__cuda_0)
# 15 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
{__device_stub__Z7ahe_GPUPh( __cuda_0);
# 21 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
}
# 1 "ahe_gpu_kernels.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T4) {  __nv_dummy_param_ref(__T4); __nv_save_fatbinhandle_for_managed_rt(__T4); __cudaRegisterEntry(__T4, ((void ( *)(unsigned char *))ahe_GPU), _Z7ahe_GPUPh, (-1)); __cudaRegisterVariable(__T4, __shadow_var(width,::width), 0, 4UL, 1, 0); __cudaRegisterVariable(__T4, __shadow_var(heigth,::heigth), 0, 4UL, 1, 0); __cudaRegisterVariable(__T4, __shadow_var(pixels_per_tile,::pixels_per_tile), 0, 4UL, 1, 0); __cudaRegisterVariable(__T4, __shadow_var(ntiles_x,::ntiles_x), 0, 4UL, 1, 0); __cudaRegisterVariable(__T4, __shadow_var(ntiles_y,::ntiles_y), 0, 4UL, 1, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
