#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "ahe_gpu_kernels.fatbin.c"
extern void __device_stub__Z11ahe_get_PDFPhPi(unsigned char *, int *);
extern void __device_stub__Z16ahe_get_mappingsPiPh(int *, unsigned char *);
extern void __device_stub__Z12ahe_equalizePhS_S_(unsigned char *, unsigned char *, unsigned char *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z11ahe_get_PDFPhPi(unsigned char *__par0, int *__par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(unsigned char *, int *))ahe_get_PDF)));}
# 41 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
void ahe_get_PDF( unsigned char *__cuda_0,int *__cuda_1)
# 41 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
{__device_stub__Z11ahe_get_PDFPhPi( __cuda_0,__cuda_1);
# 51 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
}
# 1 "ahe_gpu_kernels.cudafe1.stub.c"
void __device_stub__Z16ahe_get_mappingsPiPh( int *__par0,  unsigned char *__par1) {  __cudaLaunchPrologue(2); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaLaunch(((char *)((void ( *)(int *, unsigned char *))ahe_get_mappings))); }
# 70 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
void ahe_get_mappings( int *__cuda_0,unsigned char *__cuda_1)
# 70 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
{__device_stub__Z16ahe_get_mappingsPiPh( __cuda_0,__cuda_1);
# 93 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
}
# 1 "ahe_gpu_kernels.cudafe1.stub.c"
void __device_stub__Z12ahe_equalizePhS_S_( unsigned char *__par0,  unsigned char *__par1,  unsigned char *__par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(unsigned char *, unsigned char *, unsigned char *))ahe_equalize))); }
# 106 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
void ahe_equalize( unsigned char *__cuda_0,unsigned char *__cuda_1,unsigned char *__cuda_2)
# 106 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
{__device_stub__Z12ahe_equalizePhS_S_( __cuda_0,__cuda_1,__cuda_2);
# 164 "/home/aditya/Desktop/GPU_computing/Assignments/Assignment_1/src/ahe_gpu_kernels.cu"
}
# 1 "ahe_gpu_kernels.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T6) {  __nv_dummy_param_ref(__T6); __nv_save_fatbinhandle_for_managed_rt(__T6); __cudaRegisterEntry(__T6, ((void ( *)(unsigned char *, unsigned char *, unsigned char *))ahe_equalize), _Z12ahe_equalizePhS_S_, (-1)); __cudaRegisterEntry(__T6, ((void ( *)(int *, unsigned char *))ahe_get_mappings), _Z16ahe_get_mappingsPiPh, (-1)); __cudaRegisterEntry(__T6, ((void ( *)(unsigned char *, int *))ahe_get_PDF), _Z11ahe_get_PDFPhPi, (-1)); __cudaRegisterVariable(__T6, __shadow_var(Width,::Width), 0, 4UL, 1, 0); __cudaRegisterVariable(__T6, __shadow_var(Heigth,::Heigth), 0, 4UL, 1, 0); __cudaRegisterVariable(__T6, __shadow_var(Ntiles_x,::Ntiles_x), 0, 4UL, 1, 0); __cudaRegisterVariable(__T6, __shadow_var(Ntiles_y,::Ntiles_y), 0, 4UL, 1, 0); __cudaRegisterVariable(__T6, __shadow_var(Numtiles,::Numtiles), 0, 4UL, 1, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
