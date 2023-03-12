#ifndef RECOIL_RANS_DECODER_AVX2_32X32_H
#define RECOIL_RANS_DECODER_AVX2_32X32_H

#include "recoil/simd/rans_decoder_avx2_32x8n.h"

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity>
    class RansDecoder_AVX2_32x32 : public RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, 32> {
    public:
        using MyBase = RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, 32>;
        using MyRansDecoder = typename MyBase::MyRansDecoder;
        using MyBase::MyBase;

        const size_t NInterleaved = 32;

    protected:
        size_t decodeAligned(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset, const size_t count, std::vector<ValueType> &result) override {
            auto resultOffset = result.end() - result.begin();
            result.resize(resultOffset + count);

            u32x8 ransSimds[4];
            this->createRansSimds(ransSimds);

            const u32x8 cdfOffsets = u32x8_wrapper::setAll(cdfOffset);
            const u32x8 lutOffsets = u32x8_wrapper::setAll(lutOffset);

            u32x8 rans0 = ransSimds[0];
            u32x8 rans1 = ransSimds[1];
            u32x8 rans2 = ransSimds[2];
            u32x8 rans3 = ransSimds[3];

            auto completedCount = 0;
            for (; completedCount + NInterleaved <= count; completedCount += NInterleaved) {
                auto prob0 = this->getProbabilities(rans0);
                auto prob1 = this->getProbabilities(rans1);
                auto prob2 = this->getProbabilities(rans2);
                auto prob3 = this->getProbabilities(rans3);

                auto [sym0, start0, freq0] = this->symbolLookupAvx.getSymbolInfo(cdfOffsets, lutOffsets, prob0);
                auto [sym1, start1, freq1] = this->symbolLookupAvx.getSymbolInfo(cdfOffsets, lutOffsets, prob1);
                auto [sym2, start2, freq2] = this->symbolLookupAvx.getSymbolInfo(cdfOffsets, lutOffsets, prob2);
                auto [sym3, start3, freq3] = this->symbolLookupAvx.getSymbolInfo(cdfOffsets, lutOffsets, prob3);

                this->advanceSymbol(rans0, prob0, start0, freq0);
                this->advanceSymbol(rans1, prob1, start1, freq1);
                this->advanceSymbol(rans2, prob2, start2, freq2);
                this->advanceSymbol(rans3, prob3, start3, freq3);

                this->renormSimd(rans0);
                this->renormSimd(rans1);
                this->renormSimd(rans2);
                this->renormSimd(rans3);

                auto sym01 = _mm256_permute4x64_epi64(_mm256_packus_epi32(sym0, sym1), 0xd8);
                sym01 = _mm256_packus_epi16(sym01, sym01);
                *reinterpret_cast<uint64_t*>(&result[completedCount]) = _mm256_extract_epi64(sym01, 0);
                *reinterpret_cast<uint64_t*>(&result[completedCount + 8]) = _mm256_extract_epi64(sym01, 2);

                auto sym23 = _mm256_permute4x64_epi64(_mm256_packus_epi32(sym2, sym3), 0xd8);
                sym23 = _mm256_packus_epi16(sym23, sym23);
                *reinterpret_cast<uint64_t*>(&result[completedCount + 16]) = _mm256_extract_epi64(sym23, 0);
                *reinterpret_cast<uint64_t*>(&result[completedCount + 24]) = _mm256_extract_epi64(sym23, 2);
            }

            ransSimds[0] = rans0;
            ransSimds[1] = rans1;
            ransSimds[2] = rans2;
            ransSimds[3] = rans3;
            this->writeBackRansSimds(ransSimds);

            result.resize(resultOffset + completedCount);

            return completedCount;
        }
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity>
    RansDecoder_AVX2_32x32(std::span<uint16_t>,
                           std::array<Rans<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, 32>,
                           const CdfLutPool<uint16_t, ValueType, ProbBits, LutGranularity>&)
    -> RansDecoder_AVX2_32x32<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
}

#endif //RECOIL_RANS_DECODER_AVX2_32X32_H
