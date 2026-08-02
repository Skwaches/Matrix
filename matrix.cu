#include "matrix.cuh"
#include "kernel.cuh"
#include <cstdio>

void GPU_addition_D(std::vector<float>& A, const float B, std::vector<float> &C){
	// Allocate data
	int elements = A.size();
	size_t size = sizeof(float) * A.size();
	allocate(size, 0, size);
	
	// Copy data
	CUDA_CHECK(cudaMemcpy(data_A, A.data(), size, cudaMemcpyHostToDevice));

	// Call Kernel
	int blocks = (threadsPerBlock + elements - 1)/threadsPerBlock;
	addition_kernel_D<<<blocks, threadsPerBlock>>>(data_A, B, data_C, elements);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	//Copy Answer
	CUDA_CHECK(cudaMemcpy( C.data(), data_C, size, cudaMemcpyDeviceToHost));
}
void GPU_multiplication_D(std::vector<float>& A, const float B, std::vector<float> &C){
	// Allocate data
	int elements = A.size();
	size_t size = sizeof(float) * A.size();
	allocate(size, 0, size);
	// Copy data
	CUDA_CHECK(cudaMemcpy(data_A, A.data(), size, cudaMemcpyHostToDevice));
	// Kernel Call
	int blocks = (threadsPerBlock + elements - 1)/threadsPerBlock;
	multiplication_kernel_D<<<blocks, threadsPerBlock>>>(data_A, B, data_C, elements);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	//Copy Answer
	CUDA_CHECK(cudaMemcpy( C.data(), data_C, size, cudaMemcpyDeviceToHost));
}

//Perform size checks elsewhere!
void GPU_addition(std::vector<float>& A, const std::vector<float>& B, std::vector<float> &C){
	// Allocate data
	int elements = A.size();
	size_t size = sizeof(float) * A.size();
	allocate(size, size, size);

	// Copy data
	CUDA_CHECK(cudaMemcpy(data_A, A.data(), size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(data_B, B.data(), size, cudaMemcpyHostToDevice));

	// Call Kernel
	int blocks = (threadsPerBlock + A.size() - 1)/threadsPerBlock;
	addition_kernel<<<blocks, threadsPerBlock>>>(data_A, data_B, data_C, elements);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	//Copy Answer
	CUDA_CHECK(cudaMemcpy( C.data(), data_C, size, cudaMemcpyDeviceToHost));
}
void GPU_multiplication(
		std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
		int rows,
		int columns,
		int shared
		){
	// Allocate data
	size_t A_size = sizeof(float) * A.size();
	size_t B_size = sizeof(float) * B.size();
	size_t C_size = sizeof(float) * rows * columns;
	allocate(A_size, B_size, C_size);

	// Copy data
	CUDA_CHECK(cudaMemcpy(data_A, A.data(), A_size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(data_B, B.data(), B_size, cudaMemcpyHostToDevice));
	// CUDA_CHECK(cudaMemset(data_C, 0, C_size)); // Clearing in the compute kernel should be a bit faster

	// Call Kernel
	int blocks = (threadsPerBlock + rows * columns - 1)/threadsPerBlock;
	multiplication_kernel<<<blocks, threadsPerBlock>>>(data_A, data_B, data_C, rows*columns, columns, shared);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	//Copy Answer
	CUDA_CHECK(cudaMemcpy( C.data(), data_C, C_size, cudaMemcpyDeviceToHost));
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
