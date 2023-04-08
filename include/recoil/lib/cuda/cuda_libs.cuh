#ifndef RECOIL_CUDA_LIBS_H
#define RECOIL_CUDA_LIBS_H

#include "macros.h"
#include <stdexcept>

namespace Recoil {
    template<size_t NInterleaved, size_t NThreads>
    CUDA_DEVICE inline unsigned int getSplitId() {
        return blockIdx.x * (NThreads / NInterleaved) + threadIdx.x / NInterleaved;
    }

    template<size_t NInterleaved>
    CUDA_DEVICE inline unsigned int getDecoderId() {
        return threadIdx.x % NInterleaved;
    }

    CUDA_DEVICE inline unsigned getLaneMaskLe() {
        unsigned mask;
        asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
        return mask;
    }

    inline void cudaCheck(cudaError_t code) {
        if (code != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + std::string(cudaGetErrorString(code)));
        }
    }

    template<typename T>
    T *allocAndCopyToGpu(const T *hostPtr, const size_t size) {
        CUDA_DEVICE_PTR T *devicePtr;
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&devicePtr), size));
        cudaCheck(cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice));

        return devicePtr;
    }

    template<typename T>
    T *allocAndCopyToGpu(const std::span<T> span) {
        return allocAndCopyToGpu(span.data(), span.size_bytes());
    }

    inline int estimateMaxOccupancy(const void *func, int nThreads) {
        int maxBlocks;
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, func, nThreads, 0));

        int deviceID;
        cudaDeviceProp props;

        cudaGetDevice(&deviceID);
        cudaGetDeviceProperties(&props, deviceID);
        return maxBlocks * props.multiProcessorCount;
    }

}

#endif //RECOIL_CUDA_LIBS_H
