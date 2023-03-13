#ifndef RECOIL_RANS_DECODER_AVX2_32X8N_H
#define RECOIL_RANS_DECODER_AVX2_32X8N_H

#include "recoil/simd/rans_decoder_avx_base.h"
#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/symbol_lookup/simd/symbol_lookup_avx2_32x8.h"
#include "recoil/lib/simd/avx2_permute.h"
#include <x86intrin.h>
#include <vector>
#include <span>
#include <bit>

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity, size_t NInterleaved>
    class RansDecoder_AVX2_32x8n : public RansDecoder_AVXBase<
            uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, LutGranularity, NInterleaved, u32x8_wrapper,
            SymbolLookup_AVX2_32x8<ValueType, ProbBits, LutGranularity>> {
        using MyBase = RansDecoder_AVXBase<
                uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, LutGranularity, NInterleaved, u32x8_wrapper,
                SymbolLookup_AVX2_32x8<ValueType, ProbBits, LutGranularity>>;
        using SimdArrayType = typename MyBase::SimdArrayType;
        using MyBase::RansBatchSize, MyBase::RansStepCount;

        static constexpr size_t WriteBits = 16;
    public:
        using MyBase::MyBase;
    protected:
        [[nodiscard]] inline u32x8 getProbabilities(const u32x8 ransSimd) const override {
            const u32x8 probabilityMask = _mm256_set1_epi32((1 << ProbBits) - 1);
            return _mm256_and_si256(ransSimd, probabilityMask);
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

        inline void writeResult(const u32x8 symbolsSimd, ValueType *ptr) override {
            auto sym = _mm256_packus_epi32(symbolsSimd, symbolsSimd);
            sym = _mm256_permute4x64_epi64(sym, 0xd8);
            if constexpr (sizeof(ValueType) == 1) { // uint8_t
                sym = _mm256_packus_epi16(sym, sym);
                *reinterpret_cast<uint64_t*>(ptr) = _mm256_extract_epi64(sym, 0);
            } else if constexpr (sizeof(ValueType) == 2) { // uint16_t
                *reinterpret_cast<uint64_t*>(ptr) = _mm256_extract_epi64(sym, 0);
                *reinterpret_cast<uint64_t*>(ptr + 4) = _mm256_extract_epi64(sym, 1);
            } else {
                []<bool flag = false>() { static_assert(flag, "Unsupported value type"); }();
            }
        }

        inline void writeResult(const u32x8 symbolsSimd1, const u32x8 symbolsSimd2, ValueType *ptr) override {
            auto sym = _mm256_packus_epi32(symbolsSimd1, symbolsSimd2);
            sym = _mm256_permute4x64_epi64(sym, 0xd8);
            if constexpr (sizeof(ValueType) == 1) { // uint8_t
                sym = _mm256_packus_epi16(sym, sym);
                *reinterpret_cast<uint64_t*>(ptr) = _mm256_extract_epi64(sym, 0);
                *reinterpret_cast<uint64_t*>(ptr + 8) = _mm256_extract_epi64(sym, 2);
            } else if constexpr (sizeof(ValueType) == 2) { // uint16_t
                *reinterpret_cast<uint64_t*>(ptr) = _mm256_extract_epi64(sym, 0);
                *reinterpret_cast<uint64_t*>(ptr + 4) = _mm256_extract_epi64(sym, 1);
                *reinterpret_cast<uint64_t*>(ptr + 8) = _mm256_extract_epi64(sym, 2);
                *reinterpret_cast<uint64_t*>(ptr + 12) = _mm256_extract_epi64(sym, 3);
            } else {
                []<bool flag = false>() { static_assert(flag, "Unsupported value type"); }();
            }
        }
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity, size_t NInterleaved>
    RansDecoder_AVX2_32x8n(std::span<uint16_t>,
                           std::array<Rans<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, NInterleaved>,
                           const CdfLutPool<uint16_t, ValueType, ProbBits, LutGranularity>&)
            -> RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, NInterleaved>;
}

#endif //RECOIL_RANS_DECODER_AVX2_32X8N_H
