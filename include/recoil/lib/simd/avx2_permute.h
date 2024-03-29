#ifndef RECOIL_AVX2_PERMUTE_H
#define RECOIL_AVX2_PERMUTE_H

#include <cstdint>
#include <x86intrin.h>

namespace Recoil::AVX2Permute {
    namespace {
        const int _ = 9;
    }

    alignas(32) static const uint32_t PermuteTable[256][8] = {
            { _,_,_,_,_,_,_,_,},
            { 7,_,_,_,_,_,_,_,},
            { _,7,_,_,_,_,_,_,},
            { 7,6,_,_,_,_,_,_,},
            { _,_,7,_,_,_,_,_,},
            { 7,_,6,_,_,_,_,_,},
            { _,7,6,_,_,_,_,_,},
            { 7,6,5,_,_,_,_,_,},
            { _,_,_,7,_,_,_,_,},
            { 7,_,_,6,_,_,_,_,},
            { _,7,_,6,_,_,_,_,},
            { 7,6,_,5,_,_,_,_,},
            { _,_,7,6,_,_,_,_,},
            { 7,_,6,5,_,_,_,_,},
            { _,7,6,5,_,_,_,_,},
            { 7,6,5,4,_,_,_,_,},
            { _,_,_,_,7,_,_,_,},
            { 7,_,_,_,6,_,_,_,},
            { _,7,_,_,6,_,_,_,},
            { 7,6,_,_,5,_,_,_,},
            { _,_,7,_,6,_,_,_,},
            { 7,_,6,_,5,_,_,_,},
            { _,7,6,_,5,_,_,_,},
            { 7,6,5,_,4,_,_,_,},
            { _,_,_,7,6,_,_,_,},
            { 7,_,_,6,5,_,_,_,},
            { _,7,_,6,5,_,_,_,},
            { 7,6,_,5,4,_,_,_,},
            { _,_,7,6,5,_,_,_,},
            { 7,_,6,5,4,_,_,_,},
            { _,7,6,5,4,_,_,_,},
            { 7,6,5,4,3,_,_,_,},
            { _,_,_,_,_,7,_,_,},
            { 7,_,_,_,_,6,_,_,},
            { _,7,_,_,_,6,_,_,},
            { 7,6,_,_,_,5,_,_,},
            { _,_,7,_,_,6,_,_,},
            { 7,_,6,_,_,5,_,_,},
            { _,7,6,_,_,5,_,_,},
            { 7,6,5,_,_,4,_,_,},
            { _,_,_,7,_,6,_,_,},
            { 7,_,_,6,_,5,_,_,},
            { _,7,_,6,_,5,_,_,},
            { 7,6,_,5,_,4,_,_,},
            { _,_,7,6,_,5,_,_,},
            { 7,_,6,5,_,4,_,_,},
            { _,7,6,5,_,4,_,_,},
            { 7,6,5,4,_,3,_,_,},
            { _,_,_,_,7,6,_,_,},
            { 7,_,_,_,6,5,_,_,},
            { _,7,_,_,6,5,_,_,},
            { 7,6,_,_,5,4,_,_,},
            { _,_,7,_,6,5,_,_,},
            { 7,_,6,_,5,4,_,_,},
            { _,7,6,_,5,4,_,_,},
            { 7,6,5,_,4,3,_,_,},
            { _,_,_,7,6,5,_,_,},
            { 7,_,_,6,5,4,_,_,},
            { _,7,_,6,5,4,_,_,},
            { 7,6,_,5,4,3,_,_,},
            { _,_,7,6,5,4,_,_,},
            { 7,_,6,5,4,3,_,_,},
            { _,7,6,5,4,3,_,_,},
            { 7,6,5,4,3,2,_,_,},
            { _,_,_,_,_,_,7,_,},
            { 7,_,_,_,_,_,6,_,},
            { _,7,_,_,_,_,6,_,},
            { 7,6,_,_,_,_,5,_,},
            { _,_,7,_,_,_,6,_,},
            { 7,_,6,_,_,_,5,_,},
            { _,7,6,_,_,_,5,_,},
            { 7,6,5,_,_,_,4,_,},
            { _,_,_,7,_,_,6,_,},
            { 7,_,_,6,_,_,5,_,},
            { _,7,_,6,_,_,5,_,},
            { 7,6,_,5,_,_,4,_,},
            { _,_,7,6,_,_,5,_,},
            { 7,_,6,5,_,_,4,_,},
            { _,7,6,5,_,_,4,_,},
            { 7,6,5,4,_,_,3,_,},
            { _,_,_,_,7,_,6,_,},
            { 7,_,_,_,6,_,5,_,},
            { _,7,_,_,6,_,5,_,},
            { 7,6,_,_,5,_,4,_,},
            { _,_,7,_,6,_,5,_,},
            { 7,_,6,_,5,_,4,_,},
            { _,7,6,_,5,_,4,_,},
            { 7,6,5,_,4,_,3,_,},
            { _,_,_,7,6,_,5,_,},
            { 7,_,_,6,5,_,4,_,},
            { _,7,_,6,5,_,4,_,},
            { 7,6,_,5,4,_,3,_,},
            { _,_,7,6,5,_,4,_,},
            { 7,_,6,5,4,_,3,_,},
            { _,7,6,5,4,_,3,_,},
            { 7,6,5,4,3,_,2,_,},
            { _,_,_,_,_,7,6,_,},
            { 7,_,_,_,_,6,5,_,},
            { _,7,_,_,_,6,5,_,},
            { 7,6,_,_,_,5,4,_,},
            { _,_,7,_,_,6,5,_,},
            { 7,_,6,_,_,5,4,_,},
            { _,7,6,_,_,5,4,_,},
            { 7,6,5,_,_,4,3,_,},
            { _,_,_,7,_,6,5,_,},
            { 7,_,_,6,_,5,4,_,},
            { _,7,_,6,_,5,4,_,},
            { 7,6,_,5,_,4,3,_,},
            { _,_,7,6,_,5,4,_,},
            { 7,_,6,5,_,4,3,_,},
            { _,7,6,5,_,4,3,_,},
            { 7,6,5,4,_,3,2,_,},
            { _,_,_,_,7,6,5,_,},
            { 7,_,_,_,6,5,4,_,},
            { _,7,_,_,6,5,4,_,},
            { 7,6,_,_,5,4,3,_,},
            { _,_,7,_,6,5,4,_,},
            { 7,_,6,_,5,4,3,_,},
            { _,7,6,_,5,4,3,_,},
            { 7,6,5,_,4,3,2,_,},
            { _,_,_,7,6,5,4,_,},
            { 7,_,_,6,5,4,3,_,},
            { _,7,_,6,5,4,3,_,},
            { 7,6,_,5,4,3,2,_,},
            { _,_,7,6,5,4,3,_,},
            { 7,_,6,5,4,3,2,_,},
            { _,7,6,5,4,3,2,_,},
            { 7,6,5,4,3,2,1,_,},
            { _,_,_,_,_,_,_,7,},
            { 7,_,_,_,_,_,_,6,},
            { _,7,_,_,_,_,_,6,},
            { 7,6,_,_,_,_,_,5,},
            { _,_,7,_,_,_,_,6,},
            { 7,_,6,_,_,_,_,5,},
            { _,7,6,_,_,_,_,5,},
            { 7,6,5,_,_,_,_,4,},
            { _,_,_,7,_,_,_,6,},
            { 7,_,_,6,_,_,_,5,},
            { _,7,_,6,_,_,_,5,},
            { 7,6,_,5,_,_,_,4,},
            { _,_,7,6,_,_,_,5,},
            { 7,_,6,5,_,_,_,4,},
            { _,7,6,5,_,_,_,4,},
            { 7,6,5,4,_,_,_,3,},
            { _,_,_,_,7,_,_,6,},
            { 7,_,_,_,6,_,_,5,},
            { _,7,_,_,6,_,_,5,},
            { 7,6,_,_,5,_,_,4,},
            { _,_,7,_,6,_,_,5,},
            { 7,_,6,_,5,_,_,4,},
            { _,7,6,_,5,_,_,4,},
            { 7,6,5,_,4,_,_,3,},
            { _,_,_,7,6,_,_,5,},
            { 7,_,_,6,5,_,_,4,},
            { _,7,_,6,5,_,_,4,},
            { 7,6,_,5,4,_,_,3,},
            { _,_,7,6,5,_,_,4,},
            { 7,_,6,5,4,_,_,3,},
            { _,7,6,5,4,_,_,3,},
            { 7,6,5,4,3,_,_,2,},
            { _,_,_,_,_,7,_,6,},
            { 7,_,_,_,_,6,_,5,},
            { _,7,_,_,_,6,_,5,},
            { 7,6,_,_,_,5,_,4,},
            { _,_,7,_,_,6,_,5,},
            { 7,_,6,_,_,5,_,4,},
            { _,7,6,_,_,5,_,4,},
            { 7,6,5,_,_,4,_,3,},
            { _,_,_,7,_,6,_,5,},
            { 7,_,_,6,_,5,_,4,},
            { _,7,_,6,_,5,_,4,},
            { 7,6,_,5,_,4,_,3,},
            { _,_,7,6,_,5,_,4,},
            { 7,_,6,5,_,4,_,3,},
            { _,7,6,5,_,4,_,3,},
            { 7,6,5,4,_,3,_,2,},
            { _,_,_,_,7,6,_,5,},
            { 7,_,_,_,6,5,_,4,},
            { _,7,_,_,6,5,_,4,},
            { 7,6,_,_,5,4,_,3,},
            { _,_,7,_,6,5,_,4,},
            { 7,_,6,_,5,4,_,3,},
            { _,7,6,_,5,4,_,3,},
            { 7,6,5,_,4,3,_,2,},
            { _,_,_,7,6,5,_,4,},
            { 7,_,_,6,5,4,_,3,},
            { _,7,_,6,5,4,_,3,},
            { 7,6,_,5,4,3,_,2,},
            { _,_,7,6,5,4,_,3,},
            { 7,_,6,5,4,3,_,2,},
            { _,7,6,5,4,3,_,2,},
            { 7,6,5,4,3,2,_,1,},
            { _,_,_,_,_,_,7,6,},
            { 7,_,_,_,_,_,6,5,},
            { _,7,_,_,_,_,6,5,},
            { 7,6,_,_,_,_,5,4,},
            { _,_,7,_,_,_,6,5,},
            { 7,_,6,_,_,_,5,4,},
            { _,7,6,_,_,_,5,4,},
            { 7,6,5,_,_,_,4,3,},
            { _,_,_,7,_,_,6,5,},
            { 7,_,_,6,_,_,5,4,},
            { _,7,_,6,_,_,5,4,},
            { 7,6,_,5,_,_,4,3,},
            { _,_,7,6,_,_,5,4,},
            { 7,_,6,5,_,_,4,3,},
            { _,7,6,5,_,_,4,3,},
            { 7,6,5,4,_,_,3,2,},
            { _,_,_,_,7,_,6,5,},
            { 7,_,_,_,6,_,5,4,},
            { _,7,_,_,6,_,5,4,},
            { 7,6,_,_,5,_,4,3,},
            { _,_,7,_,6,_,5,4,},
            { 7,_,6,_,5,_,4,3,},
            { _,7,6,_,5,_,4,3,},
            { 7,6,5,_,4,_,3,2,},
            { _,_,_,7,6,_,5,4,},
            { 7,_,_,6,5,_,4,3,},
            { _,7,_,6,5,_,4,3,},
            { 7,6,_,5,4,_,3,2,},
            { _,_,7,6,5,_,4,3,},
            { 7,_,6,5,4,_,3,2,},
            { _,7,6,5,4,_,3,2,},
            { 7,6,5,4,3,_,2,1,},
            { _,_,_,_,_,7,6,5,},
            { 7,_,_,_,_,6,5,4,},
            { _,7,_,_,_,6,5,4,},
            { 7,6,_,_,_,5,4,3,},
            { _,_,7,_,_,6,5,4,},
            { 7,_,6,_,_,5,4,3,},
            { _,7,6,_,_,5,4,3,},
            { 7,6,5,_,_,4,3,2,},
            { _,_,_,7,_,6,5,4,},
            { 7,_,_,6,_,5,4,3,},
            { _,7,_,6,_,5,4,3,},
            { 7,6,_,5,_,4,3,2,},
            { _,_,7,6,_,5,4,3,},
            { 7,_,6,5,_,4,3,2,},
            { _,7,6,5,_,4,3,2,},
            { 7,6,5,4,_,3,2,1,},
            { _,_,_,_,7,6,5,4,},
            { 7,_,_,_,6,5,4,3,},
            { _,7,_,_,6,5,4,3,},
            { 7,6,_,_,5,4,3,2,},
            { _,_,7,_,6,5,4,3,},
            { 7,_,6,_,5,4,3,2,},
            { _,7,6,_,5,4,3,2,},
            { 7,6,5,_,4,3,2,1,},
            { _,_,_,7,6,5,4,3,},
            { 7,_,_,6,5,4,3,2,},
            { _,7,_,6,5,4,3,2,},
            { 7,6,_,5,4,3,2,1,},
            { _,_,7,6,5,4,3,2,},
            { 7,_,6,5,4,3,2,1,},
            { _,7,6,5,4,3,2,1,},
            { 7,6,5,4,3,2,1,0,},
    };

    inline __m256i getPermuteOffsets(size_t idx) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(PermuteTable[idx]));
    }
}

#endif //RECOIL_AVX2_PERMUTE_H
