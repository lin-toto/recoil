#ifndef RECOIL_RANS_DECODER_AVX512_32X16N_H
#define RECOIL_RANS_DECODER_AVX512_32X16N_H

#include "recoil/simd/rans_decoder_avx_base.h"
#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/symbol_lookup/simd/symbol_lookup_avx512_32x16.h"
#include "recoil/lib/simd/avx2_permute.h"
#include <x86intrin.h>
#include <vector>
#include <span>
#include <bit>

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity, size_t NInterleaved>
    class RansDecoder_AVX512_32x16n : public RansDecoder_AVXBase<
            uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, LutGranularity, NInterleaved, u32x16_wrapper,
            SymbolLookup_AVX512_32x16<ValueType, ProbBits, LutGranularity>> {
        using MyBase = RansDecoder_AVXBase<
                uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, LutGranularity, NInterleaved, u32x16_wrapper,
                SymbolLookup_AVX512_32x16<ValueType, ProbBits, LutGranularity>>;
        using SimdArrayType = typename MyBase::SimdArrayType;
        using MyBase::RansBatchSize, MyBase::RansStepCount;

        static constexpr size_t WriteBits = 16;
    public:
        using MyBase::MyBase;
    protected:
        [[nodiscard]] inline u32x16 getProbabilities(const u32x16 ransSimd) const override {
            const u32x16 probabilityMask = _mm512_set1_epi32((1 << ProbBits) - 1);
            return _mm512_and_si512(ransSimd, probabilityMask);
        }

        inline void advanceSymbol(u32x16 &ransSimd, const u32x16 lastProbabilities,
                           const u32x16 lastStarts, const u32x16 lastFrequencies) override {
            // Advance Symbols
            ransSimd = _mm512_mullo_epi32(_mm512_srli_epi32(ransSimd, ProbBits), lastFrequencies);
            ransSimd = _mm512_add_epi32(ransSimd, lastProbabilities);
            ransSimd = _mm512_sub_epi32(ransSimd, lastStarts);
        }

        inline void renormSimd(u32x16 &ransSimd) override {
            const auto reverseMask = _mm512_set_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);

            auto renormMask = _mm512_cmpgt_epu32_mask(_mm512_set1_epi32(RenormLowerBound), ransSimd);

            auto renormCount = std::popcount(static_cast<unsigned int>(renormMask));
            auto bitstreamPtr = reinterpret_cast<const __m256i*>(&(*this->bitstreamReverseIt) - RansBatchSize + 1);

            u32x16 nextBitstream = _mm512_permutexvar_epi32(reverseMask, _mm512_cvtepu16_epi32(_mm256_loadu_si256(bitstreamPtr)));
            u32x16 nextStates = _mm512_maskz_expand_epi32(renormMask, nextBitstream);

            ransSimd = _mm512_or_si512(_mm512_mask_slli_epi32(ransSimd, renormMask, ransSimd, WriteBits), nextStates);

            this->bitstreamReverseIt += renormCount;
        }

        inline void writeResult(const u32x16 symbolsSimd, ValueType *ptr) override {
            if constexpr (sizeof(ValueType) == 1) {
                auto sym = _mm512_cvtepi32_epi8(symbolsSimd);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), sym);
            } else {
                auto sym = _mm512_cvtepi32_epi16(symbolsSimd);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), sym);
            }
        }
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity, size_t NInterleaved>
    RansDecoder_AVX512_32x16n(std::span<uint16_t>,
                              std::array<Rans<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, NInterleaved>,
                              const CdfLutPool<uint16_t, ValueType, ProbBits, LutGranularity>&)
            -> RansDecoder_AVX512_32x16n<ValueType, ProbBits, RenormLowerBound, LutGranularity, NInterleaved>;
}

#endif //RECOIL_RANS_DECODER_AVX512_32X16N_H
