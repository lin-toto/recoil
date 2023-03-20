#ifndef RECOIL_RANS_SPLIT_ENCODER_H
#define RECOIL_RANS_SPLIT_ENCODER_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_encoder.h"
#include <vector>
#include <numeric>
#include <algorithm>

namespace {
    template<class T>
    T saveDiv(T a, T b) {
        return (a + b - 1) / b;
    }
}

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved>
    class RansSplitEncoder {
        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        // TODO: allow any class derived from RansEncoder, from a template parameter (we may have an AVX encoder in the future)
        using MyRansEncoder = RansEncoder<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved, true>;
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyRansSplitsMetadata = RansSplitsMetadata<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        explicit RansSplitEncoder(
                std::array<MyRans, NInterleaved> rans,  const MyCdfLutPool& pool): encoder(std::move(rans), pool) {}

        inline MyRansEncoder& getEncoder() { return encoder; }

        std::pair<MyRansCodedData, MyRansSplitsMetadata> flushSplits(size_t nSplits, SplitStrategy strategy = HeuristicSymbolCount) {
            encoder.encodeAll();

            MyRansCodedData data;
            MyRansSplitsMetadata metadata;
            switch (strategy) {
                case HeuristicSymbolCount:
                    std::tie(data, metadata) = flushSplits_heuristicSymbolCount(nSplits);
                    break;
                case EqualBitstreamLength:
                    std::tie(data, metadata) = flushSplits_equalBitstreamLength(nSplits);
                    break;
            }

            encoder.reset();
            return std::make_pair(data, metadata);
        }
    protected:
        MyRansEncoder encoder;

        /*
         * Simple heuristic strategy to try to assign close symbol counts to each split.
         */
        std::pair<MyRansCodedData, MyRansSplitsMetadata> flushSplits_heuristicSymbolCount(size_t nSplits) {
            std::array<size_t, NInterleaved> splitZoneEncoderCount{};
            std::vector<std::pair<size_t, size_t>> bestSplitPoints(
                    nSplits,
                    std::make_pair(0, std::numeric_limits<size_t>::max()));

            MyRansCodedData result{
                    encoder.symbolBuffer.size(), std::move(encoder.bitstream), std::move(encoder.rans)};

            auto targetSymbolCountPerSplit = saveDiv(encoder.symbolBuffer.size(), nSplits);

            auto stateFrontIt = encoder.intermediateStates.begin();
            auto stateRearIt = encoder.intermediateStates.begin();
            auto currentSplitId = nSplits - 1;
            for (bool initial = true;
                 stateRearIt != encoder.intermediateStates.end(); stateRearIt++) {
                splitZoneEncoderCount[stateRearIt->encoderId()]++;

                // Only check for potential split points at potentially optimal positions.
                bool potentialSplitPoint = false;
                if (initial) {
                    // The first few bytes may not have enough encoder renormalizations to become a split zone.
                    if (std::transform_reduce(
                            splitZoneEncoderCount.begin(), splitZoneEncoderCount.end(), 0, std::plus{},
                            [](auto val) { return val >= 1 ? 1 : 0; }) != NInterleaved) {
                        continue;
                    } else {
                        initial = false;
                        potentialSplitPoint = true;
                    }
                } else {
                    while (splitZoneEncoderCount[stateFrontIt->encoderId()] > 1) {
                        potentialSplitPoint = true;
                        splitZoneEncoderCount[stateFrontIt->encoderId()]--;
                        stateFrontIt++;
                    }
                }

                if (potentialSplitPoint) {
                    auto splitZoneSymbolCount = NInterleaved * (stateFrontIt->symbolGroupId() + 1 - stateRearIt->symbolGroupId());
                    auto cutPosition = std::distance(encoder.intermediateStates.begin(), stateRearIt);

                    int64_t targetSymbolId = targetSymbolCountPerSplit * currentSplitId;
                    auto heuristic = std::abs(static_cast<int64_t>(stateRearIt->symbolId) - targetSymbolId) + std::abs(static_cast<int64_t>(stateRearIt->symbolId + splitZoneSymbolCount) - targetSymbolId);
                    if (heuristic < bestSplitPoints[currentSplitId].second)
                        bestSplitPoints[currentSplitId] = std::make_pair(cutPosition, heuristic);

                    if (currentSplitId > 1) {
                        int64_t nextTargetSymbolId = targetSymbolCountPerSplit * (currentSplitId - 1);
                        if (std::abs(static_cast<int64_t>(stateRearIt->symbolId) - targetSymbolId) > std::abs(static_cast<int64_t>(stateRearIt->symbolId) - nextTargetSymbolId)) {
                            currentSplitId--;
                        }
                    }
                }
            }

            std::vector<size_t> splitPoints(nSplits);
            std::transform(bestSplitPoints.begin(), bestSplitPoints.end(), splitPoints.begin(), [](auto v) { return v.first; });
            auto metadata = buildSplitsMetadata(splitPoints, HeuristicSymbolCount, result);

            return std::make_pair(result, metadata);
        }

        std::pair<MyRansCodedData, MyRansSplitsMetadata> flushSplits_equalBitstreamLength(size_t nSplits) {
            MyRansCodedData result{
                encoder.symbolBuffer.size(), std::move(encoder.bitstream), std::move(encoder.rans)};

            auto targetLengthPerSplit = saveDiv(result.bitstream.size(), nSplits);
            std::vector<size_t> splitPoints(nSplits);
            for (auto splitId = 1; splitId < nSplits; splitId++) {
                splitPoints[splitId] = (nSplits - splitId) * targetLengthPerSplit;
            }
            auto metadata = buildSplitsMetadata(splitPoints, EqualBitstreamLength, result);

            return std::make_pair(result, metadata);
        }

        MyRansSplitsMetadata buildSplitsMetadata(const std::vector<size_t>& splitPoints, SplitStrategy strategy, const MyRansCodedData& result) {
            const auto nSplits = splitPoints.size();

            MyRansSplitsMetadata metadata{ strategy, {} };
            metadata.splits.resize(nSplits);

            metadata.splits[0] = { result.bitstream.size() - 1, result.finalRans, {} };
            for (auto splitId = 1; splitId < nSplits; splitId++) {
                auto splitPoint = splitPoints[splitId];
                auto splitPointIt = encoder.intermediateStates.begin() + splitPoint;
                metadata.splits[splitId].cutPosition = splitPoint;

                size_t foundEncoderCount = 0;
                while (foundEncoderCount != NInterleaved) {
                    if (!metadata.splits[splitId].startSymbolGroupIds[splitPointIt->encoderId()]) {
                        metadata.splits[splitId].intermediateRans[splitPointIt->encoderId()].state = splitPointIt->intermediateState;
                        metadata.splits[splitId].startSymbolGroupIds[splitPointIt->encoderId()] = splitPointIt->symbolGroupId();
                        foundEncoderCount++;
                    }
                    splitPointIt--;
                }
            }

            return metadata;
        }
    };
}

#endif //RECOIL_RANS_SPLIT_ENCODER_H
