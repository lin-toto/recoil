#ifndef RECOIL_AVX2_PERMUTE_H
#define RECOIL_AVX2_PERMUTE_H

#include <cstdint>
#include <x86intrin.h>

namespace Recoil::AVX2Permute {
    namespace {
        const int _ = 9;
    }

    static const uint32_t PermuteTable[256][8] = {
            {_, _, _, _, _, _, _, _,},
            {_, _, _, _, _, _, _, 0,},
            {_, _, _, _, _, _, _, 1,},
            {_, _, _, _, _, _, 0, 1,},
            {_, _, _, _, _, _, _, 2,},
            {_, _, _, _, _, _, 0, 2,},
            {_, _, _, _, _, _, 1, 2,},
            {_, _, _, _, _, 0, 1, 2,},
            {_, _, _, _, _, _, _, 3,},
            {_, _, _, _, _, _, 0, 3,},
            {_, _, _, _, _, _, 1, 3,},
            {_, _, _, _, _, 0, 1, 3,},
            {_, _, _, _, _, _, 2, 3,},
            {_, _, _, _, _, 0, 2, 3,},
            {_, _, _, _, _, 1, 2, 3,},
            {_, _, _, _, 0, 1, 2, 3,},
            {_, _, _, _, _, _, _, 4,},
            {_, _, _, _, _, _, 0, 4,},
            {_, _, _, _, _, _, 1, 4,},
            {_, _, _, _, _, 0, 1, 4,},
            {_, _, _, _, _, _, 2, 4,},
            {_, _, _, _, _, 0, 2, 4,},
            {_, _, _, _, _, 1, 2, 4,},
            {_, _, _, _, 0, 1, 2, 4,},
            {_, _, _, _, _, _, 3, 4,},
            {_, _, _, _, _, 0, 3, 4,},
            {_, _, _, _, _, 1, 3, 4,},
            {_, _, _, _, 0, 1, 3, 4,},
            {_, _, _, _, _, 2, 3, 4,},
            {_, _, _, _, 0, 2, 3, 4,},
            {_, _, _, _, 1, 2, 3, 4,},
            {_, _, _, 0, 1, 2, 3, 4,},
            {_, _, _, _, _, _, _, 5,},
            {_, _, _, _, _, _, 0, 5,},
            {_, _, _, _, _, _, 1, 5,},
            {_, _, _, _, _, 0, 1, 5,},
            {_, _, _, _, _, _, 2, 5,},
            {_, _, _, _, _, 0, 2, 5,},
            {_, _, _, _, _, 1, 2, 5,},
            {_, _, _, _, 0, 1, 2, 5,},
            {_, _, _, _, _, _, 3, 5,},
            {_, _, _, _, _, 0, 3, 5,},
            {_, _, _, _, _, 1, 3, 5,},
            {_, _, _, _, 0, 1, 3, 5,},
            {_, _, _, _, _, 2, 3, 5,},
            {_, _, _, _, 0, 2, 3, 5,},
            {_, _, _, _, 1, 2, 3, 5,},
            {_, _, _, 0, 1, 2, 3, 5,},
            {_, _, _, _, _, _, 4, 5,},
            {_, _, _, _, _, 0, 4, 5,},
            {_, _, _, _, _, 1, 4, 5,},
            {_, _, _, _, 0, 1, 4, 5,},
            {_, _, _, _, _, 2, 4, 5,},
            {_, _, _, _, 0, 2, 4, 5,},
            {_, _, _, _, 1, 2, 4, 5,},
            {_, _, _, 0, 1, 2, 4, 5,},
            {_, _, _, _, _, 3, 4, 5,},
            {_, _, _, _, 0, 3, 4, 5,},
            {_, _, _, _, 1, 3, 4, 5,},
            {_, _, _, 0, 1, 3, 4, 5,},
            {_, _, _, _, 2, 3, 4, 5,},
            {_, _, _, 0, 2, 3, 4, 5,},
            {_, _, _, 1, 2, 3, 4, 5,},
            {_, _, 0, 1, 2, 3, 4, 5,},
            {_, _, _, _, _, _, _, 6,},
            {_, _, _, _, _, _, 0, 6,},
            {_, _, _, _, _, _, 1, 6,},
            {_, _, _, _, _, 0, 1, 6,},
            {_, _, _, _, _, _, 2, 6,},
            {_, _, _, _, _, 0, 2, 6,},
            {_, _, _, _, _, 1, 2, 6,},
            {_, _, _, _, 0, 1, 2, 6,},
            {_, _, _, _, _, _, 3, 6,},
            {_, _, _, _, _, 0, 3, 6,},
            {_, _, _, _, _, 1, 3, 6,},
            {_, _, _, _, 0, 1, 3, 6,},
            {_, _, _, _, _, 2, 3, 6,},
            {_, _, _, _, 0, 2, 3, 6,},
            {_, _, _, _, 1, 2, 3, 6,},
            {_, _, _, 0, 1, 2, 3, 6,},
            {_, _, _, _, _, _, 4, 6,},
            {_, _, _, _, _, 0, 4, 6,},
            {_, _, _, _, _, 1, 4, 6,},
            {_, _, _, _, 0, 1, 4, 6,},
            {_, _, _, _, _, 2, 4, 6,},
            {_, _, _, _, 0, 2, 4, 6,},
            {_, _, _, _, 1, 2, 4, 6,},
            {_, _, _, 0, 1, 2, 4, 6,},
            {_, _, _, _, _, 3, 4, 6,},
            {_, _, _, _, 0, 3, 4, 6,},
            {_, _, _, _, 1, 3, 4, 6,},
            {_, _, _, 0, 1, 3, 4, 6,},
            {_, _, _, _, 2, 3, 4, 6,},
            {_, _, _, 0, 2, 3, 4, 6,},
            {_, _, _, 1, 2, 3, 4, 6,},
            {_, _, 0, 1, 2, 3, 4, 6,},
            {_, _, _, _, _, _, 5, 6,},
            {_, _, _, _, _, 0, 5, 6,},
            {_, _, _, _, _, 1, 5, 6,},
            {_, _, _, _, 0, 1, 5, 6,},
            {_, _, _, _, _, 2, 5, 6,},
            {_, _, _, _, 0, 2, 5, 6,},
            {_, _, _, _, 1, 2, 5, 6,},
            {_, _, _, 0, 1, 2, 5, 6,},
            {_, _, _, _, _, 3, 5, 6,},
            {_, _, _, _, 0, 3, 5, 6,},
            {_, _, _, _, 1, 3, 5, 6,},
            {_, _, _, 0, 1, 3, 5, 6,},
            {_, _, _, _, 2, 3, 5, 6,},
            {_, _, _, 0, 2, 3, 5, 6,},
            {_, _, _, 1, 2, 3, 5, 6,},
            {_, _, 0, 1, 2, 3, 5, 6,},
            {_, _, _, _, _, 4, 5, 6,},
            {_, _, _, _, 0, 4, 5, 6,},
            {_, _, _, _, 1, 4, 5, 6,},
            {_, _, _, 0, 1, 4, 5, 6,},
            {_, _, _, _, 2, 4, 5, 6,},
            {_, _, _, 0, 2, 4, 5, 6,},
            {_, _, _, 1, 2, 4, 5, 6,},
            {_, _, 0, 1, 2, 4, 5, 6,},
            {_, _, _, _, 3, 4, 5, 6,},
            {_, _, _, 0, 3, 4, 5, 6,},
            {_, _, _, 1, 3, 4, 5, 6,},
            {_, _, 0, 1, 3, 4, 5, 6,},
            {_, _, _, 2, 3, 4, 5, 6,},
            {_, _, 0, 2, 3, 4, 5, 6,},
            {_, _, 1, 2, 3, 4, 5, 6,},
            {_, 0, 1, 2, 3, 4, 5, 6,},
            {_, _, _, _, _, _, _, 7,},
            {_, _, _, _, _, _, 0, 7,},
            {_, _, _, _, _, _, 1, 7,},
            {_, _, _, _, _, 0, 1, 7,},
            {_, _, _, _, _, _, 2, 7,},
            {_, _, _, _, _, 0, 2, 7,},
            {_, _, _, _, _, 1, 2, 7,},
            {_, _, _, _, 0, 1, 2, 7,},
            {_, _, _, _, _, _, 3, 7,},
            {_, _, _, _, _, 0, 3, 7,},
            {_, _, _, _, _, 1, 3, 7,},
            {_, _, _, _, 0, 1, 3, 7,},
            {_, _, _, _, _, 2, 3, 7,},
            {_, _, _, _, 0, 2, 3, 7,},
            {_, _, _, _, 1, 2, 3, 7,},
            {_, _, _, 0, 1, 2, 3, 7,},
            {_, _, _, _, _, _, 4, 7,},
            {_, _, _, _, _, 0, 4, 7,},
            {_, _, _, _, _, 1, 4, 7,},
            {_, _, _, _, 0, 1, 4, 7,},
            {_, _, _, _, _, 2, 4, 7,},
            {_, _, _, _, 0, 2, 4, 7,},
            {_, _, _, _, 1, 2, 4, 7,},
            {_, _, _, 0, 1, 2, 4, 7,},
            {_, _, _, _, _, 3, 4, 7,},
            {_, _, _, _, 0, 3, 4, 7,},
            {_, _, _, _, 1, 3, 4, 7,},
            {_, _, _, 0, 1, 3, 4, 7,},
            {_, _, _, _, 2, 3, 4, 7,},
            {_, _, _, 0, 2, 3, 4, 7,},
            {_, _, _, 1, 2, 3, 4, 7,},
            {_, _, 0, 1, 2, 3, 4, 7,},
            {_, _, _, _, _, _, 5, 7,},
            {_, _, _, _, _, 0, 5, 7,},
            {_, _, _, _, _, 1, 5, 7,},
            {_, _, _, _, 0, 1, 5, 7,},
            {_, _, _, _, _, 2, 5, 7,},
            {_, _, _, _, 0, 2, 5, 7,},
            {_, _, _, _, 1, 2, 5, 7,},
            {_, _, _, 0, 1, 2, 5, 7,},
            {_, _, _, _, _, 3, 5, 7,},
            {_, _, _, _, 0, 3, 5, 7,},
            {_, _, _, _, 1, 3, 5, 7,},
            {_, _, _, 0, 1, 3, 5, 7,},
            {_, _, _, _, 2, 3, 5, 7,},
            {_, _, _, 0, 2, 3, 5, 7,},
            {_, _, _, 1, 2, 3, 5, 7,},
            {_, _, 0, 1, 2, 3, 5, 7,},
            {_, _, _, _, _, 4, 5, 7,},
            {_, _, _, _, 0, 4, 5, 7,},
            {_, _, _, _, 1, 4, 5, 7,},
            {_, _, _, 0, 1, 4, 5, 7,},
            {_, _, _, _, 2, 4, 5, 7,},
            {_, _, _, 0, 2, 4, 5, 7,},
            {_, _, _, 1, 2, 4, 5, 7,},
            {_, _, 0, 1, 2, 4, 5, 7,},
            {_, _, _, _, 3, 4, 5, 7,},
            {_, _, _, 0, 3, 4, 5, 7,},
            {_, _, _, 1, 3, 4, 5, 7,},
            {_, _, 0, 1, 3, 4, 5, 7,},
            {_, _, _, 2, 3, 4, 5, 7,},
            {_, _, 0, 2, 3, 4, 5, 7,},
            {_, _, 1, 2, 3, 4, 5, 7,},
            {_, 0, 1, 2, 3, 4, 5, 7,},
            {_, _, _, _, _, _, 6, 7,},
            {_, _, _, _, _, 0, 6, 7,},
            {_, _, _, _, _, 1, 6, 7,},
            {_, _, _, _, 0, 1, 6, 7,},
            {_, _, _, _, _, 2, 6, 7,},
            {_, _, _, _, 0, 2, 6, 7,},
            {_, _, _, _, 1, 2, 6, 7,},
            {_, _, _, 0, 1, 2, 6, 7,},
            {_, _, _, _, _, 3, 6, 7,},
            {_, _, _, _, 0, 3, 6, 7,},
            {_, _, _, _, 1, 3, 6, 7,},
            {_, _, _, 0, 1, 3, 6, 7,},
            {_, _, _, _, 2, 3, 6, 7,},
            {_, _, _, 0, 2, 3, 6, 7,},
            {_, _, _, 1, 2, 3, 6, 7,},
            {_, _, 0, 1, 2, 3, 6, 7,},
            {_, _, _, _, _, 4, 6, 7,},
            {_, _, _, _, 0, 4, 6, 7,},
            {_, _, _, _, 1, 4, 6, 7,},
            {_, _, _, 0, 1, 4, 6, 7,},
            {_, _, _, _, 2, 4, 6, 7,},
            {_, _, _, 0, 2, 4, 6, 7,},
            {_, _, _, 1, 2, 4, 6, 7,},
            {_, _, 0, 1, 2, 4, 6, 7,},
            {_, _, _, _, 3, 4, 6, 7,},
            {_, _, _, 0, 3, 4, 6, 7,},
            {_, _, _, 1, 3, 4, 6, 7,},
            {_, _, 0, 1, 3, 4, 6, 7,},
            {_, _, _, 2, 3, 4, 6, 7,},
            {_, _, 0, 2, 3, 4, 6, 7,},
            {_, _, 1, 2, 3, 4, 6, 7,},
            {_, 0, 1, 2, 3, 4, 6, 7,},
            {_, _, _, _, _, 5, 6, 7,},
            {_, _, _, _, 0, 5, 6, 7,},
            {_, _, _, _, 1, 5, 6, 7,},
            {_, _, _, 0, 1, 5, 6, 7,},
            {_, _, _, _, 2, 5, 6, 7,},
            {_, _, _, 0, 2, 5, 6, 7,},
            {_, _, _, 1, 2, 5, 6, 7,},
            {_, _, 0, 1, 2, 5, 6, 7,},
            {_, _, _, _, 3, 5, 6, 7,},
            {_, _, _, 0, 3, 5, 6, 7,},
            {_, _, _, 1, 3, 5, 6, 7,},
            {_, _, 0, 1, 3, 5, 6, 7,},
            {_, _, _, 2, 3, 5, 6, 7,},
            {_, _, 0, 2, 3, 5, 6, 7,},
            {_, _, 1, 2, 3, 5, 6, 7,},
            {_, 0, 1, 2, 3, 5, 6, 7,},
            {_, _, _, _, 4, 5, 6, 7,},
            {_, _, _, 0, 4, 5, 6, 7,},
            {_, _, _, 1, 4, 5, 6, 7,},
            {_, _, 0, 1, 4, 5, 6, 7,},
            {_, _, _, 2, 4, 5, 6, 7,},
            {_, _, 0, 2, 4, 5, 6, 7,},
            {_, _, 1, 2, 4, 5, 6, 7,},
            {_, 0, 1, 2, 4, 5, 6, 7,},
            {_, _, _, 3, 4, 5, 6, 7,},
            {_, _, 0, 3, 4, 5, 6, 7,},
            {_, _, 1, 3, 4, 5, 6, 7,},
            {_, 0, 1, 3, 4, 5, 6, 7,},
            {_, _, 2, 3, 4, 5, 6, 7,},
            {_, 0, 2, 3, 4, 5, 6, 7,},
            {_, 1, 2, 3, 4, 5, 6, 7,},
            {0, 1, 2, 3, 4, 5, 6, 7,},
    };

    inline __m256i getPermuteOffsets(size_t idx) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(PermuteTable[idx]));
    }
}

#endif //RECOIL_AVX2_PERMUTE_H
