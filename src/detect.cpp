#include "matrix.hpp"
#include "matrix.cuh"
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
bool cudaAvailable(void){
    int deviceCount = 0;
    cudaError_t error= cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return deviceCount > 0;
}
const bool GPU = cudaAvailable();
#else
void quitCUDA(){return;}
#endif
