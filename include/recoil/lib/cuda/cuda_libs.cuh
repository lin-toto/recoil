#ifndef RECOIL_CUDA_LIBS_H
#define RECOIL_CUDA_LIBS_H

#include "macros.h"

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
}

#endif //RECOIL_CUDA_LIBS_H
