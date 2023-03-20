#ifndef RECOIL_RANS_DECODER_AVX512_32X32_H
#define RECOIL_RANS_DECODER_AVX512_32X32_H

#include "recoil/simd/rans_decoder_avx512_32x16n.h"

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity>
    class RansDecoder_AVX512_32x32 : public RansDecoder_AVX512_32x16n<ValueType, ProbBits, RenormLowerBound, LutGranularity, 32> {
    public:
        using MyBase = RansDecoder_AVX512_32x16n<ValueType, ProbBits, RenormLowerBound, LutGranularity, 32>;
        using MyRansDecoder = typename MyBase::MyRansDecoder;
        using MyBase::MyBase;

        const size_t NInterleaved = 32;

    protected:
        void decodeOnceAligned(const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets,
                               u32x16 ransSimds[], const std::span<ValueType> output)  override {
            auto prob0 = this->getProbabilities(ransSimds[0]);
            auto prob1 = this->getProbabilities(ransSimds[1]);

            auto cdfOffsetsSimd = u32x16_wrapper::toSimd(cdfOffsets.data());
            auto lutOffsetsSimd = u32x16_wrapper::toSimd(lutOffsets.data());
            auto [sym0, start0, freq0] = this->symbolLookupAvx.getSymbolInfo(cdfOffsetsSimd, lutOffsetsSimd, prob0);

            cdfOffsetsSimd = u32x16_wrapper::toSimd(cdfOffsets.data() + 16);
            lutOffsetsSimd = u32x16_wrapper::toSimd(lutOffsets.data() + 16);
            auto [sym1, start1, freq1] = this->symbolLookupAvx.getSymbolInfo(cdfOffsetsSimd, lutOffsetsSimd, prob0);

            this->advanceSymbol(ransSimds[0], prob0, start0, freq0);
            this->advanceSymbol(ransSimds[1], prob1, start1, freq1);

            this->renormSimd(ransSimds[0]);
            this->renormSimd(ransSimds[1]);

            this->writeResult(sym0, output.data());
            this->writeResult(sym1, output.data());
        }
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint32_t RenormLowerBound, uint8_t LutGranularity>
    RansDecoder_AVX512_32x32(std::span<uint16_t>,
                              std::array<Rans<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, 16>,
                              const CdfLutPool<uint16_t, ValueType, ProbBits, LutGranularity>&)
    -> RansDecoder_AVX512_32x32<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
}

#endif //RECOIL_RANS_DECODER_AVX2_32X16_H
