#pragma once

#include <iostream>
#define CUDA_CHECK(expression) do {\
	cudaError_t success = expression;\
	if( success != cudaSuccess )\
		std::cout << "Cuda Runtime Error: " << __FILE__ << \
		__LINE__ << ' ' << cudaGetErrorString(success) << std::endl;\
	}while(0)

extern float *data_A;
extern float *data_B;
extern float *data_C;
extern int threadsPerBlock;

void allocate(size_t size_A, size_t size_B, size_t size_C);
void quitCUDA();
__global__ void addition_kernel_D( float* A, const float B, float* C, int size);
__global__ void multiplication_kernel_D( float* A, const float B, float* C, int size);
__global__ void addition_kernel( float* A, const float* B, float* C, int size);
__global__ void multiplication_kernel( 
		float* A, const float* B, float* C, 
		int size, // First Matrix rows
		int columns, // Second Matrix columns
		int shared // Columns of the first & Rows of the second
		);
