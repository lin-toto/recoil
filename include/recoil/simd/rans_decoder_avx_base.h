#ifndef RECOIL_RANS_DECODER_AVX_BASE_H
#define RECOIL_RANS_DECODER_AVX_BASE_H

#include "recoil/rans_decoder.h"
#include <array>

namespace Recoil {
    template<typename RansStateType, typename RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t NInterleaved, typename SimdDataTypeWrapper>
    class RansDecoder_AVXBase : public RansDecoder<
            RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved> {
    protected:
        using MyRansDecoder = RansDecoder<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using SimdDataType = typename SimdDataTypeWrapper::SimdDataType;

        static constexpr size_t RansBatchSize = sizeof(SimdDataType) / sizeof(RansStateType);
        static constexpr size_t RansStepCount = NInterleaved / RansBatchSize;

        using SimdArrayType = std::array<RansStateType, RansBatchSize>;

        static_assert(NInterleaved % RansBatchSize == 0, "AVX decoder must work on RansPerBatchxN streams");
        static_assert(MyRansDecoder::MyRans::oneShotRenorm, "Only one shot renorm decoders are supported by AVX decoder");
    public:
        using MyRansDecoder::MyRansDecoder;

        std::vector<ValueType> decode(const Cdf cdf) {
            std::vector<ValueType> result;
            // TODO

            return result;
        }

        std::vector<ValueType> decode(const Cdf cdf, const size_t count) {
            std::vector<ValueType> result;
            result.reserve(count);
            size_t completedCount = 0;

            {
                // Step 1: decode initial unaligned parts so that ransIt is now at beginning
                auto unalignedCount = std::min(static_cast<size_t>(this->rans.end() - this->ransIt), count);
                if (unalignedCount != NInterleaved) { // If equal to NInterleaved, it is at beginning; no action needed
                    auto initialUnalignedResult = MyRansDecoder::decode(cdf, unalignedCount);
                    completedCount += unalignedCount;
                    result.insert(result.end(), initialUnalignedResult.begin(), initialUnalignedResult.end());
                }

                if (completedCount == count) [[unlikely]] return result;
            }

            {
                // Step 2: do simd rANS decoding
                std::array<Cdf, RansBatchSize> cdfs = {cdf, cdf, cdf, cdf, cdf, cdf, cdf, cdf};

                SimdDataType ransSimds[RansStepCount];
                createRansSimds(ransSimds);

                for (; completedCount + NInterleaved <= count; completedCount += NInterleaved) {
                    for (auto b = 0; b < RansStepCount; b++) {
                        auto& ransSimd = ransSimds[b];
                        auto probabilitiesSimd = getProbabilities(ransSimd);
                        auto [bypass, symbolsSimd, startsSimd, frequenciesSimd] = getSymbolsAndStartsAndFrequencies(probabilitiesSimd, cdfs);
                        // TODO: if probability is a bypass sentinel, handle as bypass symbol

                        renorm(ransSimd, probabilitiesSimd, startsSimd, frequenciesSimd);

                        auto symbols = fromSimd(symbolsSimd);
                        result.insert(result.end(), symbols.begin(), symbols.end());
                    }
                }

                writeBackRansSimds(ransSimds);
            }

            {
                // Step 3: decode final unaligned parts
                if (completedCount != count) {
                    auto finalUnalignedResult = MyRansDecoder::decode(cdf, count - completedCount);
                    result.insert(result.end(), finalUnalignedResult.begin(), finalUnalignedResult.end());
                }
            }

            return result;
        }

        std::vector<ValueType> decode(const std::span<Cdf> cdfs) {
            std::vector<ValueType> result;
            result.reserve(cdfs.size());
            // TODO

            return result;
        }
    protected:
        void createRansSimds(SimdDataType* ransSimds) {
            for (auto b = 0; b < RansStepCount; b++) {
                alignas(sizeof(SimdDataType)) SimdArrayType rans{};
                for (auto i = 0; i < RansBatchSize; i++) {
                    rans[i] = this->rans[b * RansBatchSize + i].state;
                }
                ransSimds[b] = toSimd(rans);
            }
        };

        void writeBackRansSimds(SimdDataType *ransSimds) {
            for (auto b = 0; b < RansStepCount; b++) {
                auto rans = fromSimd(ransSimds[b]);
                for (auto i = 0; i < RansBatchSize; i++) {
                    this->rans[b * RansBatchSize + i].state = rans[i];
                }
            }
        };

        auto getSymbolsAndStartsAndFrequencies(const SimdDataType probabilitiesSimd, const std::array<Cdf, RansBatchSize> &cdfs) {
            std::array<bool, RansBatchSize> bypass{};
            alignas(sizeof(SimdDataType)) SimdArrayType symbols{}, starts{}, frequencies{};
            auto probabilities = fromSimd(probabilitiesSimd);

            for (size_t i = 0; i < RansBatchSize; i++) {
                auto symbol = cdfs[i].findValue(probabilities[i]);
                if (symbol.has_value()) [[likely]] {
                    bypass[i] = false;
                    symbols[i] = symbol.value();
                    std::tie(starts[i], frequencies[i]) = cdfs[i].getStartAndFrequency(symbols[i]).value();
                } else {
                    bypass[i] = true;
                }
            }

            return std::make_tuple(bypass, toSimd(symbols), toSimd(starts), toSimd(frequencies));
        }

        virtual SimdDataType toSimd(const SimdArrayType &val) const = 0;
        virtual SimdArrayType fromSimd(SimdDataType simd) const = 0;

        virtual SimdDataType getProbabilities(SimdDataType ransSimd) const = 0;
        virtual void renorm(SimdDataType &ransSimd, SimdDataType lastProbabilities, SimdDataType lastStarts,
                            SimdDataType lastFrequencies) = 0;
    };
}

#endif //RECOIL_RANS_DECODER_AVX_BASE_H
