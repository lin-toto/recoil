#ifndef RECOIL_SYMBOL_LOOKUP_AVX512_32X16_H
#define RECOIL_SYMBOL_LOOKUP_AVX512_32X16_H

#include "recoil/symbol_lookup/simd/symbol_lookup_avx_base.h"
#include <x86intrin.h>

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity>
    class SymbolLookup_AVX512_32x16 : public SymbolLookup_AVX_Base<ValueType, ProbBits, LutGranularity, u32x16_wrapper> {
        using MyBase = SymbolLookup_AVX_Base<ValueType, ProbBits, LutGranularity, u32x16_wrapper>;
        using MyLutItem = typename MyBase::MyLutItem;
    public:
        using MyBase::MyBase;
    protected:
        [[nodiscard]] inline u32x16 valueOnlyLutLookup(const u32x16 lutOffsets, const u32x16 probabilities) const override {
            const u32x16 symbolMask = _mm512_set1_epi32((1 << (8 * sizeof(ValueType))) - 1);

            u32x16 offsets = _mm512_add_epi32(lutOffsets, probabilities);
            u32x16 rawValues = _mm512_i32gather_epi32(offsets, reinterpret_cast<const int*>(this->lutPool), sizeof(MyLutItem));
            return _mm512_and_si512(rawValues, symbolMask);
        }

        [[nodiscard]] inline typename MyBase::SymbolInfo getSymbolInfo_mixed(
                const u32x16 cdfOffsets, const u32x16 startPositions, const u32x16 probabilities) const override {
            typename MyBase::SymbolInfo symbolInfo;

            throw std::runtime_error("Not implemented for AVX512");

            /*getOneSymbolInfo_mixed<0>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<1>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<2>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<3>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<4>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<5>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<6>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<7>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<8>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<9>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<10>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<11>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<12>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<13>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<14>(symbolInfo, cdfOffsets, startPositions, probabilities);
            getOneSymbolInfo_mixed<15>(symbolInfo, cdfOffsets, startPositions, probabilities);

            return symbolInfo;*/
        }

        [[nodiscard]] inline typename MyBase::SymbolInfo getSymbolInfo_lutOnly(
                const u32x16 lutOffsets, const u32x16 probabilities) const override {
            const u32x16 symbolMask = _mm512_set1_epi32((1 << (8 * sizeof(ValueType))) - 1);
            const u32x16 cdfMask = _mm512_set1_epi32(0xffff);

            u32x16 offsets = _mm512_add_epi32(lutOffsets, probabilities);
            u32x16 startsAndFrequencies = _mm512_i32gather_epi32(offsets, reinterpret_cast<const int*>(this->lutPool), sizeof(uint64_t));

            u32x16 symbolOffsets = _mm512_add_epi32(_mm512_slli_epi32(offsets, 1), _mm512_set1_epi32(1));
            u32x16 symbols = _mm512_i32gather_epi32(symbolOffsets, reinterpret_cast<const int*>(this->lutPool), sizeof(uint64_t) / 2);
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
            u32x16 lutReadout = _mm512_i32gather_epi32(offsets, reinterpret_cast<const int*>(this->lutPool), sizeof(uint32_t));

            u32x16 symbols = _mm512_and_si512(lutReadout, symbolMask);
            u32x16 starts = _mm512_and_si512(_mm512_srli_epi32(lutReadout, 8), cdfMask);
            u32x16 frequencies = _mm512_and_si512(_mm512_srli_epi32(lutReadout, 20), cdfMask);

            return { symbols, starts, frequencies };
        }
    /*private:
        template<const int i>
        inline void getOneSymbolInfo_mixed(
                typename MyBase::SymbolInfo& symbolInfo,
                const u32x16 cdfOffsets, const u32x16 startPositions, const u32x16 probabilities) const {
            auto [value, start, frequency] = this->linearSearch(
                    _mm512_extract_epi32(cdfOffsets, i),
                    _mm512_extract_epi32(probabilities, i),
                    _mm512_extract_epi32(startPositions, i));
            symbolInfo.value = _mm512_insert_epi32(symbolInfo.value, value, i);
            symbolInfo.start = _mm512_insert_epi32(symbolInfo.start, start, i);
            symbolInfo.frequency = _mm512_insert_epi32(symbolInfo.frequency, frequency, i);
        }*/
    };
}

#endif //RECOIL_SYMBOL_LOOKUP_AVX512_32X16_H
