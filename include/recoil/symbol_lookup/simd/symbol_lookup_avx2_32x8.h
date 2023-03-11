#ifndef RECOIL_SYMBOL_LOOKUP_AVX2_32X8_H
#define RECOIL_SYMBOL_LOOKUP_AVX2_32X8_H

#include "recoil/symbol_lookup/simd/symbol_lookup_avx_base.h"
#include <x86intrin.h>

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity>
    class SymbolLookup_AVX2_32x8 : public SymbolLookup_AVX_Base<ValueType, ProbBits, LutGranularity, u32x8_wrapper> {
        using MyBase = SymbolLookup_AVX_Base<ValueType, ProbBits, LutGranularity, u32x8_wrapper>;
        using MyLutItem = typename MyBase::MyLutItem;
    public:
        using MyBase::MyBase;
    protected:
        inline typename MyBase::SymbolInfo getSymbolInfo_lutOnly(u32x8 lutOffsets, u32x8 probabilities) const override {
            const u32x8 symbolMask = _mm256_set1_epi32((1 << (8 * sizeof(ValueType))) - 1);
            const u32x8 cdfMask = _mm256_set1_epi32(0xffff);

            u32x8 offsets = _mm256_add_epi32(lutOffsets, probabilities);
            u32x8 startsAndFrequencies = _mm256_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), offsets, sizeof(uint64_t));

            u32x8 symbolOffsets = _mm256_add_epi32(_mm256_slli_epi32(offsets, 1), _mm256_set1_epi32(1));
            u32x8 symbols = _mm256_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), symbolOffsets, sizeof(uint64_t) / 2);
            symbols = _mm256_and_si256(symbols, symbolMask);

            u32x8 starts = _mm256_and_si256(startsAndFrequencies, cdfMask);
            u32x8 frequencies = _mm256_srli_epi32(startsAndFrequencies, sizeof(uint16_t) * 8);

            return { symbols, starts, frequencies };
        }

        inline typename MyBase::SymbolInfo getSymbolInfo_lutOnly_packed(u32x8 lutOffsets, u32x8 probabilities) const override {
            const u32x8 symbolMask = _mm256_set1_epi32(0xff);
            const u32x8 cdfMask = _mm256_set1_epi32(0x0fff);

            u32x8 offsets = _mm256_add_epi32(lutOffsets, probabilities);
            u32x8 lutReadout = _mm256_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), offsets, sizeof(uint32_t));

            u32x8 symbols = _mm256_and_si256(lutReadout, symbolMask);
            u32x8 starts = _mm256_and_si256(_mm256_srli_epi32(lutReadout, 8), cdfMask);
            u32x8 frequencies = _mm256_and_si256(_mm256_srli_epi32(lutReadout, 20), cdfMask);

            return { symbols, starts, frequencies };
        }
    };
}

#endif //RECOIL_SYMBOL_LOOKUP_AVX2_32X8_H
