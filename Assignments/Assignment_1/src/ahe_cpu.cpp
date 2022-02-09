#include "ahe_cpu.h"
#include <cstring>
#include <cmath>
#include<iostream>
#include<map>
#include<vector>
#include <fstream>

// Histogram equalization: https://en.wikipedia.org/wiki/Histogram_equalization
// Adaptive Histogram equalization: https://en.wikipedia.org/wiki/Adaptive_histogram_equalization

unsigned char interp2(unsigned char v00, unsigned char v01, unsigned char v10, unsigned char v11, float x_frac, float y_frac)
{
  float v0 = v00*(1 - x_frac) + v10*x_frac;
	float v1 = v01*(1 - x_frac) + v11*x_frac;
  float v = v0*(1 - y_frac) + v1*y_frac;

	if (v < 0) v = 0;
	if (v > 255) v = 255;

	return (unsigned char)(v);
}

template<typename T>
void print_mapping(T* mappings, int mapping_size, std::string filename){
	std::map<int, int> counts;
	std::cout << "Printing mappings on CPU\n";
    for(int i = 0; i < mapping_size; i++){
        int i__ = mappings[i];
        if(counts.find(i__) == counts.end()){
			counts[i__] = 1;
			continue;
		}
		counts[i__]++;
    }
	std::ofstream MyFile(filename+".txt");
	for(auto& p : counts){
		MyFile << p.first << ':' << p.second << '\n';
	}
    MyFile << std::endl;
	MyFile.close();
}

void adaptiveEqualizationCPU(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  	int pdf[256], cdf[256];
	int pixels_per_tile = TILE_SIZE_X*TILE_SIZE_Y;
	int ntiles_x = width / TILE_SIZE_X;
	int ntiles_y = height / TILE_SIZE_Y;

	std::vector<int> global_pdf;
  // Step 1: Caculate equalization mapping for each tile
	int mapping_size = ntiles_x*ntiles_y*256;
	unsigned char *mappings = new unsigned char[mapping_size];
  	for(int y = 0; y < height; y+=TILE_SIZE_Y){
		  for(int x = 0; x < width; x+=TILE_SIZE_X) {
			// Compute PDF
			memset(pdf, 0, 256*sizeof(int));
			
			// Compute frequencies
			for(int j=y; j<(y+TILE_SIZE_Y); j++) {
				for(int i=x; i<(x+TILE_SIZE_X); i++){
					pdf[img_in[i+j*width]]++;
				}
			}
						

			// Compute CDF
			cdf[0] = pdf[0];
			global_pdf.push_back(pdf[0]);
			for(int i=1; i< 256; i++){
				cdf[i] = cdf[i-1] + pdf[i];
				global_pdf.push_back(pdf[i]);
			}
			
			int cdf_min = pixels_per_tile+1; // minimum non-zero value of the CDF
			for(int i=0; i<256; i++){
				if(cdf[i] != 0) {
					cdf_min = cdf[i]; 
					break;
				}
			}
				

			// Compute Map
			int tile_i = x / TILE_SIZE_X;
			int tile_j = y / TILE_SIZE_Y;
			int offset = 256*(tile_i + tile_j*ntiles_x);
			for(int i=0; i< 256; i++){
				auto val = (255.0 * float(cdf[i] - cdf_min)/float(pixels_per_tile - cdf_min));
				mappings[i+ offset] = (unsigned char)round(val);
				// mappings[i+ offset] = (unsigned char)cdf[i];
			}
		}
	}
	print_mapping(mappings, mapping_size, "mappings_CPU");
  // Step 2: Perform adaptive equalization. For each pixel in a tile, interpolate results from neighbouring mappings
  	int tile_i0, tile_j0, tile_i1, tile_j1; // tile IDs
  	for(int y = 0; y < height; y++){
		for(int x = 0; x < width; x++)
		{
			tile_i0 = (x - TILE_SIZE_X/2) / TILE_SIZE_X;
			if(tile_i0 < 0) {
				tile_i0 = 0;
			}
			
			tile_j0 = (y - TILE_SIZE_Y/2) / TILE_SIZE_Y;
			if(tile_j0 < 0) {
				tile_j0 = 0;
			}

			tile_i1 = (x + TILE_SIZE_X/2) / TILE_SIZE_X;
			if(tile_i1 >= ntiles_x){
				tile_i1 = ntiles_x - 1;
			} 
			
			tile_j1 = (y + TILE_SIZE_Y/2) / TILE_SIZE_Y;
			if(tile_j1 >= ntiles_y) {
				tile_j1 = ntiles_y - 1;
			}

			// Find offsets to neighboring mappings. For no neighbors, set the nearest neighbor.
			int offset00 = 256*(tile_i0 + tile_j0*ntiles_x);
			int offset01 = 256*(tile_i0 + tile_j1*ntiles_x);
			int offset10 = 256*(tile_i1 + tile_j0*ntiles_x);
			int offset11 = 256*(tile_i1 + tile_j1*ntiles_x);

			// Compute 4 values and perform bilinear interpolation
      		unsigned char v00, v01, v10, v11, v;
			v00 = mappings[img_in[x+y*width] + offset00];
			v01 = mappings[img_in[x+y*width] + offset01];
			v10 = mappings[img_in[x+y*width] + offset10];
			v11 = mappings[img_in[x+y*width] + offset11];
			float x_frac = float(x - tile_i0*TILE_SIZE_X - TILE_SIZE_X/2)/float(TILE_SIZE_X);
			float y_frac = float(y - tile_j0*TILE_SIZE_Y - TILE_SIZE_Y/2)/float(TILE_SIZE_Y);

      		img_out[x+y*width] = interp2(v00, v01, v10, v11, x_frac, y_frac);
		}
  	}

	//Cleanup
	delete []mappings;
}
