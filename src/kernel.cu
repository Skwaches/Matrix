#include "kernel.cuh"
#include "matrix.cuh"

float *data_A = nullptr, 
			  *data_B = nullptr,
			  *data_C = nullptr;
static size_t space_A = 0,
			  space_B = 0,
			  space_C = 0;
int threadsPerBlock = 256;

// This ensures memory is only allocated when what is currently allocated is insufficient.
void allocate(size_t size_A, size_t size_B, size_t size_C){
	if( size_A > space_A ){
		if (data_A)
			cudaFree(data_A);
		CUDA_CHECK(cudaMalloc(&data_A,size_A));
		space_A = size_A;
	}
	if( size_B > space_B ){
		if (data_B)
			cudaFree(data_B);
		CUDA_CHECK(cudaMalloc(&data_B,size_B));
		space_B = size_B;
	}
	if( size_C > space_C ){
		if (data_C)
			cudaFree(data_C);
		CUDA_CHECK(cudaMalloc(&data_C,size_C));
		space_C = size_C;
	}
	return;
}

__global__ void addition_kernel_D( float* A, const float B, float* C, int size){
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadId < size){
		C[threadId] = A[threadId] + B;
	}
}
__global__ void multiplication_kernel_D( float* A, const float B, float* C, int size){
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadId < size){
		C[threadId] = A[threadId] * B;
	}
}

__global__ void addition_kernel( float* A, const float* B, float* C, int size){
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadId < size){
		C[threadId] = A[threadId] + B[threadId];
	}
}

__global__ void multiplication_kernel( 
		float* A, const float* B, float* C, 
		int size, // First Matrix rows
		int columns, // Second Matrix columns
		int shared // Columns of the first & Rows of the second
		){
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < size){
		int row = threadId/columns;
		int column = threadId%columns;
		C[threadId] = 0; // Initialise the value. // Slightly faster than cudaMemset.
		for (int i = 0; i < shared; i++){
			C[threadId] += A[ row * shared + i] * B[ column + i * columns];
		}
	}
}

void quitCUDA(){
	CUDA_CHECK(cudaFree(data_A));
	CUDA_CHECK(cudaFree(data_B));
	CUDA_CHECK(cudaFree(data_C));

	// Never done this before lol
	// This is disgusting.
	data_A = data_B = data_C = nullptr;
	space_A = space_B = space_C = 0;
}
