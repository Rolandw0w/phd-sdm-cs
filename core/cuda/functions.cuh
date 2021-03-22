#ifndef cuda_functions_cuh
#define cuda_functions_cuh


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <functional>

#include "../utils/utils.hpp"


#define GPUErrorCheck(ans) { GPUError((ans), __FILE__, __LINE__); }
inline void GPUError(cudaError_t code, char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU error: %s %s %d\n", cudaGetErrorString(code), file, line);
        throw std::exception(cudaGetErrorString(code));
    }
}

template<typename T>
void check_kernel_errors(T synchronize_device)
{
    if (synchronize_device)
    {
        cudaDeviceSynchronize();
    }
    GPUError(cudaGetLastError(), __FILE__, __LINE__);
}

template<typename Kernel, typename... Args>
inline void kernel_decorator(Kernel kernel, int block_count, int threads_per_block, Args&&... args)
{
    kernel<<<block_count, threads_per_block>>>(args...);
    check_kernel_errors<bool>(true);
}

template<typename T>
void cuda_malloc(T* cuda_ptr, ulong size)
{
    GPUErrorCheck(cudaMalloc((void**)&cuda_ptr, size*sizeof(T)));
}

template<typename T>
void cuda_memcpy(T* cuda_ptr, T* ptr, ulong size)
{
    GPUErrorCheck(cudaMemcpy(cuda_ptr, ptr, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cuda_free(T* cuda_ptr)
{
    GPUErrorCheck(cudaFree(cuda_ptr));
}


#endif //cuda_functions_cuh
