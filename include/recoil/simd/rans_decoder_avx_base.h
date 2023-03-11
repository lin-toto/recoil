#ifndef RECOIL_RANS_DECODER_AVX_BASE_H
#define RECOIL_RANS_DECODER_AVX_BASE_H

#include "recoil/rans_decoder.h"
#include "recoil/lib/simd/avx_datatypes.h"
#include <x86intrin.h>
#include <array>
#include <tuple>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved, SimdDataTypeWrapperConcept SimdDataTypeWrapper, typename SymbolLookupAVX>
    class RansDecoder_AVXBase : public RansDecoder<
            CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved> {
    protected:
        using MyRansDecoder = RansDecoder<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved>;
        using SimdDataType = typename SimdDataTypeWrapper::SimdDataType;
        using MyRans = typename MyRansDecoder::MyRans;
        using MyCdfLutPool = typename MyRansDecoder::MyCdfLutPool;

        static constexpr size_t RansBatchSize = sizeof(SimdDataType) / sizeof(RansStateType);
        static constexpr size_t RansStepCount = NInterleaved / RansBatchSize;

        using SimdArrayType = std::array<RansStateType, RansBatchSize>;

        static_assert(NInterleaved % RansBatchSize == 0, "AVX decoder must work on RansPerBatchxN streams");
        static_assert(MyRansDecoder::MyRans::oneShotRenorm, "Only one shot renorm decoders are supported by AVX decoder");
    public:
        RansDecoder_AVXBase(const std::span<RansBitstreamType> bitstream, std::array<MyRans, NInterleaved> rans, const MyCdfLutPool& pool)
            : symbolLookupAvx(pool), MyRansDecoder(bitstream, std::move(rans), pool) {}

        std::vector<ValueType> decode(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            throw std::runtime_error("Not implemented for AVX");
        }

        std::vector<ValueType> decode(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset, const size_t count) {
            // TODO: support LUT/CDF mixed lookup

            std::vector<ValueType> result;
            result.reserve(count);
            size_t completedCount = 0;

            {
                // Step 1: decode initial unaligned parts so that ransIt is now at beginning
                auto unalignedCount = std::min(static_cast<size_t>(this->rans.end() - this->ransIt), count);
                if (unalignedCount != NInterleaved) { // If equal to NInterleaved, it is at beginning; no action needed
                    auto initialUnalignedResult = MyRansDecoder::decode(cdfOffset, lutOffset, unalignedCount);
                    completedCount += unalignedCount;
                    result.insert(result.end(), initialUnalignedResult.begin(), initialUnalignedResult.end());
                }

                if (completedCount == count) [[unlikely]] return result;
            }

            {
                SimdDataType ransSimds[RansStepCount];
                createRansSimds(ransSimds);

                const SimdDataType cdfOffsets = SimdDataTypeWrapper::setAll(cdfOffset);
                const SimdDataType lutOffsets = SimdDataTypeWrapper::setAll(lutOffset);

                for (; completedCount + NInterleaved <= count; completedCount += NInterleaved) {
                    for (auto b = 0; b < RansStepCount; b++) {
                        auto& ransSimd = ransSimds[b];
                        auto probabilitiesSimd = getProbabilities(ransSimd);

                        auto [symbolsSimd, startsSimd, frequenciesSimd] = symbolLookupAvx.getSymbolInfo(cdfOffsets, lutOffsets, probabilitiesSimd);

                        // TODO: if probability is a bypass sentinel, handle as bypass symbol

                        advanceSymbol(ransSimd, probabilitiesSimd, startsSimd, frequenciesSimd);
                        renormSimd(ransSimd);

                        auto symbols = SimdDataTypeWrapper::fromSimd(symbolsSimd);
                        result.insert(result.end(), symbols.begin(), symbols.end());
                    }
                }

                writeBackRansSimds(ransSimds);
            }

            {
                // Step 3: decode final unaligned parts
                if (completedCount != count) {
                    auto finalUnalignedResult = MyRansDecoder::decode(cdfOffset, lutOffset, count - completedCount);
                    result.insert(result.end(), finalUnalignedResult.begin(), finalUnalignedResult.end());
                }
            }

            return result;
        }

        std::vector<ValueType> decode(const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets) {
            assert(cdfOffsets.size() == lutOffsets.size());

            std::vector<ValueType> result;
            result.reserve(cdfOffsets.size());
            // TODO

            return result;
        }
    protected:
        SymbolLookupAVX symbolLookupAvx;

        inline void createRansSimds(SimdDataType* ransSimds) {
            for (auto b = 0; b < RansStepCount; b++) {
                alignas(sizeof(SimdDataType)) SimdArrayType rans{};
                for (auto i = 0; i < RansBatchSize; i++) {
                    rans[i] = this->rans[b * RansBatchSize + i].state;
                }
                ransSimds[b] = SimdDataTypeWrapper::toSimd(rans);
            }
        };

        inline void writeBackRansSimds(SimdDataType *ransSimds) {
            for (auto b = 0; b < RansStepCount; b++) {
                auto rans = SimdDataTypeWrapper::fromSimd(ransSimds[b]);
                for (auto i = 0; i < RansBatchSize; i++) {
                    this->rans[b * RansBatchSize + i].state = rans[i];
                }
            }
        };


        virtual SimdDataType getProbabilities(SimdDataType ransSimd) const = 0;
        virtual void advanceSymbol(SimdDataType &ransSimd, SimdDataType lastProbabilities,
                                   SimdDataType lastStarts, SimdDataType lastFrequencies) = 0;
        virtual void renormSimd(SimdDataType &ransSimd) = 0;
    };
}

#endif //RECOIL_RANS_DECODER_AVX_BASE_H
