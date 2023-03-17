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
        [[nodiscard]] inline u32x8 valueOnlyLutLookup(const u32x8 lutOffsets, const u32x8 probabilities) const override {
            const u32x8 symbolMask = _mm256_set1_epi32((1 << (8 * sizeof(ValueType))) - 1);

            u32x8 offsets = _mm256_add_epi32(lutOffsets, probabilities);
            u32x8 rawValues = _mm256_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), offsets, sizeof(MyLutItem));
            return _mm256_and_si256(rawValues, symbolMask);
        }

        [[nodiscard]] inline typename MyBase::SymbolInfo getSymbolInfo_mixed(
                const u32x8 cdfOffsets, const u32x8 startPositions, const u32x8 probabilities) const override {
            typename MyBase::SymbolInfo symbolInfo;

            getOneSymbolInfo_mixed(symbolInfo, 0, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 1, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 2, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 3, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 4, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 5, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 6, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 7, cdfOffsets, startPositions, probabilities);

            return symbolInfo;
        }

        [[nodiscard]] inline typename MyBase::SymbolInfo getSymbolInfo_lutOnly(
                const u32x8 lutOffsets, const u32x8 probabilities) const override {
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

        [[nodiscard]]  inline typename MyBase::SymbolInfo getSymbolInfo_lutOnly_packed(
                const u32x8 lutOffsets, const u32x8 probabilities) const override {
            const u32x8 symbolMask = _mm256_set1_epi32(0xff);
            const u32x8 cdfMask = _mm256_set1_epi32(0x0fff);

            u32x8 offsets = _mm256_add_epi32(lutOffsets, probabilities);
            u32x8 lutReadout = _mm256_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), offsets, sizeof(uint32_t));

            u32x8 symbols = _mm256_and_si256(lutReadout, symbolMask);
            u32x8 starts = _mm256_and_si256(_mm256_srli_epi32(lutReadout, 8), cdfMask);
            u32x8 frequencies = _mm256_and_si256(_mm256_srli_epi32(lutReadout, 20), cdfMask);

            return { symbols, starts, frequencies };
        }
    private:
        inline void getOneSymbolInfo_mixed(
                typename MyBase::SymbolInfo& symbolInfo, const int i,
                const u32x8 cdfOffsets, const u32x8 startPositions, const u32x8 probabilities) const {
            auto [value, start, frequency] = this->linearSearch(
                    _mm256_extract_epi32(cdfOffsets, i),
                    _mm256_extract_epi32(probabilities, i),
                    _mm256_extract_epi32(startPositions, i));
            symbolInfo.value = _mm256_insert_epi32(symbolInfo.value, value, i);
            symbolInfo.start = _mm256_insert_epi32(symbolInfo.start, start, i);
            symbolInfo.frequency = _mm256_insert_epi32(symbolInfo.frequency, frequency, i);
        }
    };
}

#endif //RECOIL_SYMBOL_LOOKUP_AVX2_32X8_H
