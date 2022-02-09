#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <sstream>
#include "stb_image.h"
#include "stb_image_write.h"
#include "ahe_cpu.h"
#include "ahe_gpu.h"

#define ENABLE_TIMER 1
#define ENABLE_SAVE_IMAGE 1

void verifyCPUGPU(unsigned char* out_cpu, unsigned char* out_gpu, size_t length);

using namespace std;

int main(int argc, char **argv) {
  if(argc < 2) {
    cout<<"Usage: " << argv[0] << " <image_file>\n";
    return 1;
    }
  std::cout << "Tile Size : " << TILE_SIZE_X << std::endl;
  // Read input image
  int width, height, nchannels;
	cout<<"Reading "<<argv[1]<<"... "<<flush;
  unsigned char *img_in = stbi_load(argv[1], &width, &height, &nchannels, 0);
  cout<<"Width: "<< width << " Height: " << height <<" Channels: "<< nchannels << "\n";
  if(nchannels != 1) {
    cout<<"Only single channel (8-bit) grascale images are supported! Exiting...\n";
    return 1;
  }

  // Create output array for image
  unsigned char *img_out_cpu = new unsigned char[width*height];
  unsigned char *img_out_gpu = new unsigned char[width*height];

  // Perform Adaptive histogram equalization on CPU and save image
  memset(img_out_cpu, 0, width*height);
	cout<<"Performing adaptive equalization on CPU... "<<flush;
#if ENABLE_TIMER
	struct timespec start_cpu, end_cpu;
	float msecs_cpu;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);
#endif
  adaptiveEqualizationCPU(img_in, img_out_cpu, width, height);
#if ENABLE_TIMER
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_cpu);
	msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
	cout<<"done in "<<msecs_cpu<<" milliseconds.\n"<<flush;
#else
  cout<<"done.\n"<<flush;
#endif
#if ENABLE_SAVE_IMAGE
  std::stringstream ss_cpu;
  ss_cpu << TILE_SIZE_X << "out_CPU.png";
  std::string filename_cpu(ss_cpu.str());

  cout<<"Saving output to " << filename_cpu <<endl;
  
  stbi_write_png(filename_cpu.c_str(), width, height, 1, img_out_cpu, width);
	cout<<"done.\n"<<flush;
#endif

  // Perform Adaptive histogram equalization on GPU
  memset(img_out_gpu, 0, width*height);
	cout<<"Performing adaptive equalization on GPU... "<<flush;
#if ENABLE_TIMER
	cudaEvent_t start_gpu, end_gpu;
	float msecs_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&end_gpu);
	cudaEventRecord(start_gpu, 0);
#endif
  adaptiveEqualizationGPU(img_in, img_out_gpu, width, height);
#if ENABLE_TIMER
	cudaEventRecord(end_gpu, 0);
	cudaEventSynchronize(end_gpu);
	cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(end_gpu);
	cout<<"done in "<<msecs_gpu<<" milliseconds.\n";
#else
  cout<<"done.\n"<<flush;
#endif
#if ENABLE_SAVE_IMAGE
  std::stringstream ss_gpu;
  ss_gpu << TILE_SIZE_X << "out_GPU.png";
  std::string filename_gpu(ss_gpu.str());

  cout<<"Saving output to " << filename_gpu<<endl;
  
  stbi_write_png(filename_gpu.c_str(), width, height, 1, img_out_gpu, width);
	cout<<"done.\n"<<flush;
#endif
  
	//Verify GPU output
	verifyCPUGPU(img_out_cpu, img_out_gpu, width*height);

  // Cleanup and exit
  delete [] img_out_cpu;
  delete [] img_out_gpu;

  return 0;
}

void verifyCPUGPU(unsigned char* out_cpu, unsigned char* out_gpu, size_t length)
{
  cout<<"Verifying GPU output with CPU output..."<<flush;

	double rms = 0.0;
	int diff;
  double abs_diff =0.0;
	for(size_t i=0; i<length; i++) {
    diff = out_cpu[i] - out_gpu[i];
    abs_diff += std::abs(diff);
		rms += diff*diff;
	}

	rms  = sqrt(rms / length);
  abs_diff = abs_diff / length;
  cout<<" RMS error: "<< rms << endl;
  cout<<" Mean absolute pixel error: "<< abs_diff << endl;
}
