#ifndef RECOIL_SYMBOL_LOOKUP_AVX512_32X16_H
#define RECOIL_SYMBOL_LOOKUP_AVX512_32X16_H

#include "recoil/symbol_lookup/simd/symbol_lookup_avx_base.h"
#include <x86intrin.h>

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity>
    class SymbolLookup_AVX512_32x16 : public SymbolLookup_AVX_Base<ValueType, ProbBits, LutGranularity, u32x16_wrapper> {
        using MyBase = SymbolLookup_AVX_Base<ValueType, ProbBits, LutGranularity, u32x8_wrapper>;
        using MyLutItem = typename MyBase::MyLutItem;
    public:
        using MyBase::MyBase;
    protected:
        [[nodiscard]] inline u32x16 valueOnlyLutLookup(const u32x16 lutOffsets, const u32x16 probabilities) const override {
            const u32x16 symbolMask = _mm512_set1_epi32((1 << (8 * sizeof(ValueType))) - 1);

            u32x16 offsets = _mm512_add_epi32(lutOffsets, probabilities);
            u32x16 rawValues = _mm512_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), offsets, sizeof(MyLutItem));
            return _mm512_and_si512(rawValues, symbolMask);
        }

        [[nodiscard]] inline typename MyBase::SymbolInfo getSymbolInfo_mixed(
                const u32x16 cdfOffsets, const u32x16 startPositions, const u32x16 probabilities) const override {
            typename MyBase::SymbolInfo symbolInfo;

            getOneSymbolInfo_mixed(symbolInfo, 0, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 1, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 2, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 3, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 4, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 5, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 6, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 7, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 8, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 9, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 10, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 11, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 12, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 13, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 14, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed(symbolInfo, 15, cdfOffsets, startPositions, probabilities);

            return symbolInfo;
        }

        [[nodiscard]] inline typename MyBase::SymbolInfo getSymbolInfo_lutOnly(
                const u32x16 lutOffsets, const u32x16 probabilities) const override {
            const u32x16 symbolMask = _mm512_set1_epi32((1 << (8 * sizeof(ValueType))) - 1);
            const u32x16 cdfMask = _mm512_set1_epi32(0xffff);

            u32x16 offsets = _mm512_add_epi32(lutOffsets, probabilities);
            u32x16 startsAndFrequencies = _mm512_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), offsets, sizeof(uint64_t));

            u32x16 symbolOffsets = _mm512_add_epi32(_mm512_slli_epi32(offsets, 1), _mm256_set1_epi32(1));
            u32x16 symbols = _mm512_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), symbolOffsets, sizeof(uint64_t) / 2);
            symbols = _mm512_and_si512(symbols, symbolMask);

            u32x16 starts = _mm512_and_si512(startsAndFrequencies, cdfMask);
            u32x16 frequencies = _mm512_srli_epi32(startsAndFrequencies, sizeof(uint16_t) * 8);

            return { symbols, starts, frequencies };
        }

        [[nodiscard]]  inline typename MyBase::SymbolInfo getSymbolInfo_lutOnly_packed(
                const u32x16 lutOffsets, const u32x16 probabilities) const override {
            const u32x16 symbolMask = _mm512_set1_epi32(0xff);
            const u32x16 cdfMask = _mm512_set1_epi32(0x0fff);

            u32x16 offsets = _mm512_add_epi32(lutOffsets, probabilities);
            u32x16 lutReadout = _mm512_i32gather_epi32(reinterpret_cast<const int*>(this->lutPool), offsets, sizeof(uint32_t));

            u32x16 symbols = _mm512_and_si512(lutReadout, symbolMask);
            u32x16 starts = _mm512_and_si512(_mm512_srli_epi32(lutReadout, 8), cdfMask);
            u32x16 frequencies = _mm512_and_si512(_mm512_srli_epi32(lutReadout, 20), cdfMask);

            return { symbols, starts, frequencies };
        }
    private:
        inline void getOneSymbolInfo_mixed(
                typename MyBase::SymbolInfo& symbolInfo, const int i,
                const u32x16 cdfOffsets, const u32x16 startPositions, const u32x16 probabilities) const {
            auto [value, start, frequency] = this->linearSearch(
                    _mm512_extract_epi32(cdfOffsets, i),
                    _mm512_extract_epi32(probabilities, i),
                    _mm512_extract_epi32(startPositions, i));
            symbolInfo.value = _mm512_insert_epi32(symbolInfo.value, value, i);
            symbolInfo.start = _mm512_insert_epi32(symbolInfo.start, start, i);
            symbolInfo.frequency = _mm512_insert_epi32(symbolInfo.frequency, frequency, i);
        }
    };
}

#endif //RECOIL_SYMBOL_LOOKUP_AVX512_32X16_H
