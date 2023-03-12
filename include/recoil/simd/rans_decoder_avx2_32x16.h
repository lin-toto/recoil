#ifndef RECOIL_RANS_DECODER_AVX2_32X16_H
#define RECOIL_RANS_DECODER_AVX2_32X16_H

#include "recoil/simd/rans_decoder_avx2_32x8n.h"

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity>
    class RansDecoder_AVX2_32x16 : public RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, 16> {
    public:
        using MyBase = RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, 16>;
        using MyRansDecoder = typename MyBase::MyRansDecoder;
        using MyBase::MyBase;

        const size_t NInterleaved = 16;

    protected:
        size_t decodeAligned(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset,
                             const size_t count, std::vector<ValueType> &result, const size_t writeOffset) override {
            u32x8 ransSimds[2];
            this->createRansSimds(ransSimds);

            const u32x8 cdfOffsets = u32x8_wrapper::setAll(cdfOffset);
            const u32x8 lutOffsets = u32x8_wrapper::setAll(lutOffset);

            u32x8 rans0 = ransSimds[0];
            u32x8 rans1 = ransSimds[1];

            auto completedCount = 0;
            for (; completedCount + NInterleaved <= count; completedCount += NInterleaved) {
                auto prob0 = this->getProbabilities(rans0);
                auto prob1 = this->getProbabilities(rans1);

                auto [sym0, start0, freq0] = this->symbolLookupAvx.getSymbolInfo(cdfOffsets, lutOffsets, prob0);
                auto [sym1, start1, freq1] = this->symbolLookupAvx.getSymbolInfo(cdfOffsets, lutOffsets, prob1);

                this->advanceSymbol(rans0, prob0, start0, freq0);
                this->advanceSymbol(rans1, prob1, start1, freq1);

                this->renormSimd(rans0);
                this->renormSimd(rans1);

                this->writeResult(sym0, sym1, result, writeOffset + completedCount);
            }

            ransSimds[0] = rans0;
            ransSimds[1] = rans1;
            this->writeBackRansSimds(ransSimds);

            return completedCount;
        }
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity>
    RansDecoder_AVX2_32x16(std::span<uint16_t>,
                           std::array<Rans<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, 16>,
                           const CdfLutPool<uint16_t, ValueType, ProbBits, LutGranularity>&)
    -> RansDecoder_AVX2_32x16<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
}

#endif //RECOIL_RANS_DECODER_AVX2_32X16_H
