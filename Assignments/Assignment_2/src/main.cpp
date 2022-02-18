#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <ctime>
#include <float.h>

#include <cuda_runtime.h>

#include "stb_image.h"
#include "stb_image_write.h"
#include "sdt_cpu.h"
#include "sdt_gpu.h"

#define ENABLE_TIMER 1
#define ENABLE_SAVING 1

using namespace std;

void saveImage(const char *filename, float * sdt, int width, int height, unsigned char *bitmap);
void compareSDT(float *sdt1, float *sdt2, int width, int height);

int main(int argc, char **argv) {
  if(argc < 2) {
    cout<<"Usage: " << argv[0] << " <image_file>\n";
    return 1;
    }

  // Read input image
  int width, height, nchannels;
  cout<<"Reading "<<argv[1]<<"... "<<flush;
  unsigned char *img = stbi_load(argv[1], &width, &height, &nchannels, 0);
  cout<<"Width: "<< width << " Height: " << height <<" Channels: "<< nchannels << "\n";
  if(nchannels != 1) {
    cout<<"Only single channel (8-bit) grascale images are supported! Exiting...\n";
    return 1;
  }

  // Create SDT array for CPU
  float *sdt_cpu = new float[width*height];

  // Compute SDT on CPU and save image
  cout<<"Computing SDT on CPU... \n"<<flush;
#if ENABLE_TIMER
  struct timespec start_cpu, end_cpu;
  float msecs_cpu;
  clock_gettime(CLOCK_MONOTONIC, &start_cpu);
#endif
  computeSDT_CPU(img, sdt_cpu, width, height);
#if ENABLE_TIMER
  clock_gettime(CLOCK_MONOTONIC, &end_cpu);
  msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
  cout<<"\tComputation took "<<msecs_cpu<<" milliseconds.\n"<<flush;
#else
  cout<<"\tDone.\n"<<flush;
#endif

#if ENABLE_SAVING
  cout<<"\tSaving output to sdt_CPU.png... "<<flush;
  saveImage("sdt_CPU.png", sdt_cpu, width, height, img);
  cout<<"done.\n"<<flush;
#endif


  // Create SDT array for GPU
  float *sdt_gpu = new float[width*height];

  // Compute SDT on GPU
  cout<<"Computing SDT on GPU... \n"<<flush;
#if ENABLE_TIMER
  cudaEvent_t start_gpu, end_gpu;
  float msecs_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&end_gpu);
  cudaEventRecord(start_gpu, 0);
#endif
  computeSDT_GPU(img, sdt_gpu, width, height);
#if ENABLE_TIMER
  cudaEventRecord(end_gpu, 0);
  cudaEventSynchronize(end_gpu);
  cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(end_gpu);
  cout<<"\tComputation took "<<msecs_gpu<<" milliseconds.\n";
#else
  cout<<"\tDone.\n"<<flush;
#endif

#if ENABLE_SAVING
  cout<<"\tSaving output to sdt_GPU.png... "<<flush;
  saveImage("sdt_GPU.png", sdt_gpu, width, height, img);
  cout<<"done.\n"<<flush;
#endif

  // Compare outputs
  compareSDT(sdt_cpu, sdt_gpu, width, height);

  // Cleanup and exit
  delete [] sdt_cpu;
  delete [] sdt_gpu;
  delete [] img;

  return 0;
}

void saveImage(const char *filename, float * sdt, int width, int height, unsigned char *bitmap)
{
  float mind = FLT_MAX, maxd = -FLT_MAX;
	
  int sz  = width*height;
  float val;
  for(int i=0; i<sz; i++) // Find min/max of data
  {
    val  = sdt[i];
    if(val < mind) mind = val;
    if(val > maxd) maxd  = val;
  }
  unsigned char *data = new unsigned char[3*sz*sizeof(unsigned char)];
  for(int y = 0; y<height; y++) // Convert image to 24 bit
    for(int x=0; x<width; x++)
    {
      val = sdt[x + y*width];
      data[(x + y*width)*3 + 1] = 0;
      if(val<0) 
      {
        data[(x + y*width)*3 + 0] = 0;
	data[(x + y*width)*3 + 2] = 255*val/mind;
      } else {
	data[(x + y*width)*3 + 0] = 255*val/maxd;
	data[(x + y*width)*3 + 2] = 0;
      }
    }
  for(int i=0; i<sz; i++) // Mark boundary
    if(bitmap[i] == 255) {data[i*3] = 255; data[i*3+1] = 255; data[i*3+2] = 255;}

  stbi_write_png(filename, width, height, 3, data, width*3);

  delete []data;
}

void compareSDT(float *sdt1, float *sdt2, int width, int height)
{
	//Compare Mean Square Error between the two distance maps
	float mse = 0.0f;
	int sz = width*height;
	for(int i=0; i<sz; i++)
		mse += (sdt1[i] - sdt2[i])*(sdt1[i] - sdt2[i]);
	mse  = sqrtf(mse/sz);
	cout<<"CPU-GPU Mean Square Error (MSE): "<< mse <<"\n"<<flush;
}
