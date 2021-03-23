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

//#define DEBUG
#ifdef DEBUG
#define DEFAULT_SYNCHRONIZE_DEVICE true
#else
#define DEFAULT_SYNCHRONIZE_DEVICE false
#endif


#define GPUErrorCheck(ans, synchronize_device) { GPUError((ans), __FILE__, __LINE__, synchronize_device); }
inline void GPUError(cudaError_t code, std::string file, int line, bool synchronize_device = true)
{
    if (synchronize_device)
        cudaDeviceSynchronize();

    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU error: %s %s %d\n", cudaGetErrorString(code), file.data(), line);
        throw std::exception(cudaGetErrorString(code));
    }
}

template<typename Kernel, typename... Args>
inline void kernel_decorator(Kernel kernel, int block_count, int threads_per_block, bool synchronize_device, Args&&... args)
{
    kernel<<<block_count, threads_per_block>>>(args...);
    GPUError(cudaGetLastError(), __FILE__, __LINE__, synchronize_device);
}

template<typename T>
void cuda_malloc(T** cuda_ptr, ulong size, bool ignore_sizeof = false, bool synchronize_device = DEFAULT_SYNCHRONIZE_DEVICE)
{
    ulong bytes = ignore_sizeof ? size : size*sizeof(T);
    GPUErrorCheck(cudaMalloc((void**)cuda_ptr, bytes), synchronize_device);
}

template<typename T>
void cuda_memcpy_to_gpu(T* cuda_ptr, T* ptr, ulong size, bool ignore_sizeof = false, bool synchronize_device = DEFAULT_SYNCHRONIZE_DEVICE)
{
    ulong bytes = ignore_sizeof ? size : size*sizeof(T);
    GPUErrorCheck(cudaMemcpy(cuda_ptr, ptr, bytes, cudaMemcpyHostToDevice), synchronize_device);
}

template<typename T>
void cuda_memcpy_to_gpu(T* cuda_ptr, const T* ptr, ulong size, bool ignore_sizeof = false, bool synchronize_device = DEFAULT_SYNCHRONIZE_DEVICE)
{
    ulong bytes = ignore_sizeof ? size : size*sizeof(T);
    GPUErrorCheck(cudaMemcpy(cuda_ptr, ptr, bytes, cudaMemcpyHostToDevice), synchronize_device);
}

template<typename T>
void cuda_memcpy_from_gpu(T* ptr, T* cuda_ptr, ulong size, bool ignore_sizeof = false, bool synchronize_device = DEFAULT_SYNCHRONIZE_DEVICE)
{
    ulong bytes = ignore_sizeof ? size : size*sizeof(T);
    GPUErrorCheck(cudaMemcpy(ptr, cuda_ptr, bytes, cudaMemcpyDeviceToHost), synchronize_device);
}

template<typename T>
void cuda_memcpy_from_gpu(T* ptr, const T* cuda_ptr, ulong size, bool ignore_sizeof = false, bool synchronize_device = DEFAULT_SYNCHRONIZE_DEVICE)
{
    ulong bytes = ignore_sizeof ? size : size*sizeof(T);
    GPUErrorCheck(cudaMemcpy(ptr, cuda_ptr, bytes, cudaMemcpyDeviceToHost), synchronize_device);
}

template<typename T>
void cuda_free(T* cuda_ptr, bool synchronize_device = DEFAULT_SYNCHRONIZE_DEVICE)
{
    GPUErrorCheck(cudaFree(cuda_ptr), synchronize_device);
}

template<typename T>
void cuda_memset(T* cuda_ptr, int value, ulong size, bool ignore_sizeof = false, bool synchronize_device = DEFAULT_SYNCHRONIZE_DEVICE)
{
    ulong bytes = ignore_sizeof ? size : size*sizeof(T);
    GPUErrorCheck(cudaMemset(cuda_ptr, value, bytes), synchronize_device);
}


#endif //cuda_functions_cuh
