#ifndef RECOIL_CUDA_MACROS_H
#define RECOIL_CUDA_MACROS_H

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define CUDA_HOST_DEVICE __device__ __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#elif defined(__CUDACC_RTC__)
#define CUDA_HOST_DEVICE __device__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_GLOBAL
#endif

#endif //RECOIL_CUDA_MACROS_H
