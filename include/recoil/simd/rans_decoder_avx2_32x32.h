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
        void decodeOnceAligned(const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets,
                               u32x8 ransSimds[], const std::span<ValueType> output) override {
            auto prob0 = this->getProbabilities(ransSimds[0]);
            auto prob1 = this->getProbabilities(ransSimds[1]);
            auto prob2 = this->getProbabilities(ransSimds[2]);
            auto prob3 = this->getProbabilities(ransSimds[3]);

            auto cdfOffsetsSimd = u32x8_wrapper::toSimd(cdfOffsets.data());
            auto lutOffsetsSimd = u32x8_wrapper::toSimd(lutOffsets.data());
            auto [sym0, start0, freq0] = this->symbolLookupAvx.getSymbolInfo(cdfOffsetsSimd, lutOffsetsSimd, prob0);

            cdfOffsetsSimd = u32x8_wrapper::toSimd(cdfOffsets.data() + 8);
            lutOffsetsSimd = u32x8_wrapper::toSimd(lutOffsets.data() + 8);
            auto [sym1, start1, freq1] = this->symbolLookupAvx.getSymbolInfo(cdfOffsetsSimd, lutOffsetsSimd, prob1);

            cdfOffsetsSimd = u32x8_wrapper::toSimd(cdfOffsets.data() + 16);
            lutOffsetsSimd = u32x8_wrapper::toSimd(lutOffsets.data() + 16);
            auto [sym2, start2, freq2] = this->symbolLookupAvx.getSymbolInfo(cdfOffsetsSimd, lutOffsetsSimd, prob2);

            cdfOffsetsSimd = u32x8_wrapper::toSimd(cdfOffsets.data() + 24);
            lutOffsetsSimd = u32x8_wrapper::toSimd(lutOffsets.data() + 24);
            auto [sym3, start3, freq3] = this->symbolLookupAvx.getSymbolInfo(cdfOffsetsSimd, lutOffsetsSimd, prob3);

            this->advanceSymbol(ransSimds[0], prob0, start0, freq0);
            this->advanceSymbol(ransSimds[1], prob1, start1, freq1);
            this->advanceSymbol(ransSimds[2], prob2, start2, freq2);
            this->advanceSymbol(ransSimds[3], prob3, start3, freq3);

            this->renormSimd(ransSimds[0]);
            this->renormSimd(ransSimds[1]);
            this->renormSimd(ransSimds[2]);
            this->renormSimd(ransSimds[3]);

            this->writeResult(sym0, sym1, output.data());
            this->writeResult(sym2, sym3, output.data() + 16);
        }
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity>
    RansDecoder_AVX2_32x32(std::span<uint16_t>,
                           std::array<Rans<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, 32>,
                           const CdfLutPool<uint16_t, ValueType, ProbBits, LutGranularity>&)
    -> RansDecoder_AVX2_32x32<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
}

#endif //RECOIL_RANS_DECODER_AVX2_32X32_H
