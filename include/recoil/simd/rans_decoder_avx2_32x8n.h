#ifndef RECOIL_RANS_DECODER_AVX2_32X8N_H
#define RECOIL_RANS_DECODER_AVX2_32X8N_H

#include "recoil/simd/rans_decoder_avx_base.h"
#include "recoil/lib/simd/avx2_permute.h"
#include <x86intrin.h>
#include <vector>
#include <span>
#include <bit>

namespace Recoil {
    namespace {
        using u32x8 = __m256i;
        struct u32x8_wrapper {
            using SimdDataType = u32x8;
        };
    }

    template<uint8_t ProbBits, uint32_t RenormLowerBound, size_t NInterleaved>
    class RansDecoder_AVX2_32x8n : public RansDecoder_AVXBase<
            uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, NInterleaved, u32x8_wrapper> {
        using MyBase = RansDecoder_AVXBase<uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, NInterleaved, u32x8_wrapper>;
        using SimdArrayType = typename MyBase::SimdArrayType;
        using MyBase::RansBatchSize, MyBase::RansStepCount;

        static constexpr size_t WriteBits = 16;
    public:
        using MyBase::MyBase;
    protected:
        [[nodiscard]] inline u32x8 toSimd(const SimdArrayType &val) const override {
            return _mm256_load_si256(reinterpret_cast<const u32x8*>(val.begin()));
        }

        [[nodiscard]] inline SimdArrayType fromSimd(const u32x8 simd) const override {
            alignas(32) std::array<uint32_t, RansBatchSize> val;
            _mm256_store_si256(reinterpret_cast<u32x8*>(val.begin()), simd);
            return val;
        }

        [[nodiscard]] inline u32x8 getProbabilities(const u32x8 ransSimd) const override {
            const u32x8 probabilityMask = _mm256_set1_epi32((1 << ProbBits) - 1);
            return _mm256_and_si256(ransSimd, probabilityMask);
        }

        [[nodiscard]] inline std::tuple<u32x8, u32x8, u32x8> getSymbolsAndStartsAndFrequenciesSimd_staticCdf_LutOnly(
                const u32x8 probabilitiesSimd, const Cdf cdf) const override {
            const u32x8 valueMask = _mm256_set1_epi32((1 << (8 * sizeof(ValueType))) - 1);
            const u32x8 cdfMask = _mm256_set1_epi32((1 << (8 * sizeof(CdfType))) - 1);

            u32x8 symbols = _mm256_and_si256(
                    _mm256_i32gather_epi32(reinterpret_cast<const int*>(&(*cdf.lut.begin())), probabilitiesSimd, sizeof(ValueType)),
                    valueMask);

            u32x8 cdfReadout = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&(*cdf.cdf.begin())), symbols, sizeof(CdfType));
            u32x8 starts = _mm256_and_si256(cdfReadout, cdfMask);
            u32x8 nextStarts = _mm256_srli_epi32(cdfReadout, sizeof(CdfType) * 8);
            if constexpr (sizeof(cdfMask) == 1) nextStarts = _mm256_and_si256(nextStarts, cdfMask);
            u32x8 frequencies = _mm256_sub_epi32(nextStarts, starts);

            return std::make_tuple(symbols, starts, frequencies);
        }

        inline void advanceSymbol(u32x8 &ransSimd, const u32x8 lastProbabilities,
                           const u32x8 lastStarts, const u32x8 lastFrequencies) override {
            // Advance Symbols
            ransSimd = _mm256_mullo_epi32(_mm256_srli_epi32(ransSimd, ProbBits), lastFrequencies);
            ransSimd = _mm256_add_epi32(ransSimd, lastProbabilities);
            ransSimd = _mm256_sub_epi32(ransSimd, lastStarts);
        }

        inline void renormSimd(u32x8 &ransSimd) override {
            // Check renormalization flags; dirty hack because unsigned comparison is not supported in AVX2
            const u32x8 renormLowerBound = _mm256_set1_epi32(RenormLowerBound - 0x80000000);
            const u32x8 signFlag = _mm256_set1_epi32(static_cast<int>(0x80000000u));

            /*if (this->bitstreamReverseIt == this->bitstream.rend()) {
                const u32x8 lowerBound = _mm256_set1_epi32(RenormLowerBound);
                if (_mm256_movemask_ps(reinterpret_cast<__m256>(_mm256_cmpgt_epi32(_mm256_xor_si256(ransSimd, signFlag), lowerBound)))) {
                    throw DecodingReachesEndException();
                }
                return;
            }*/

            u32x8 renormMaskSimd = _mm256_cmpgt_epi32(renormLowerBound,_mm256_xor_si256(ransSimd, signFlag));

            auto renormMask = _mm256_movemask_ps(reinterpret_cast<__m256>(renormMaskSimd));
            /*            <--------------------------vv
             * Bitstream: 01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef
             * RenormCount: 2 (Expected to read:  ^^ ^^)
             * Read from ptr - 7 then set ptr -= 2
             */

            auto renormCount = std::popcount(static_cast<unsigned int>(renormMask));
            auto bitstreamPtr = reinterpret_cast<const __m128i*>(&(*this->bitstreamReverseIt) - RansBatchSize + 1);

            // Use _mm_loadu_si128 because it does not require memory alignment.
            u32x8 nextBitstream = _mm256_cvtepu16_epi32(_mm_loadu_si128(bitstreamPtr));
            u32x8 nextStates = _mm256_permutevar8x32_epi32(nextBitstream, AVX2Permute::getPermuteOffsets(renormMask));

            u32x8 renormedRans = _mm256_or_si256(_mm256_slli_epi32(ransSimd, WriteBits), nextStates);
            ransSimd = _mm256_blendv_epi8(ransSimd, renormedRans, renormMaskSimd);

            this->bitstreamReverseIt += renormCount;
        }
    };

    template<uint8_t ProbBits, uint32_t RenormLowerBound, size_t nInterleaved>
    RansDecoder_AVX2_32x8n(std::span<uint16_t>,
                           std::array<Rans<uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, nInterleaved>)
            -> RansDecoder_AVX2_32x8n<ProbBits, RenormLowerBound, nInterleaved>;
}

#endif //RECOIL_RANS_DECODER_AVX2_32X8N_H
