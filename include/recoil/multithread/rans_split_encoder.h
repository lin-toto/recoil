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
    template<std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits,
            size_t NInterleaved>
    class RansSplitEncoder {
        using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        // TODO: allow any class derived from RansEncoder, from a template parameter
        using MyRansEncoder = RansEncoder<
                RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved, true>;
        template<size_t NSplits>
        using MyRansCodedDataWithSplits = RansCodedDataWithSplits<
                RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved, NSplits>;
    public:
        explicit RansSplitEncoder(
                std::array<MyRans, NInterleaved> rans): encoder(std::move(rans)) {}

        inline MyRansEncoder& getEncoder() { return encoder; }

        template<size_t NSplits>
        MyRansCodedDataWithSplits<NSplits> flushSplits(SplitStrategy strategy = HeuristicSymbolCount) {
            encoder.encodeAll();

            MyRansCodedDataWithSplits<NSplits> result;
            switch (strategy) {
                case HeuristicSymbolCount:
                    result = flushSplits_heuristicSymbolCount<NSplits>();
                    break;
                case EqualBitstreamLength:
                    result = flushSplits_equalBitstreamLength<NSplits>();
                    break;
            }

            encoder.reset();
            return result;
        }
    protected:
        MyRansEncoder encoder;

        /*
         * Simple heuristic strategy to try to assign close symbol counts to each split.
         */
        template<size_t NSplits>
        MyRansCodedDataWithSplits<NSplits> flushSplits_heuristicSymbolCount() {
            std::array<size_t, NInterleaved> splitZoneEncoderCount{};
            std::array<std::pair<size_t, size_t>, NSplits> bestSplitPoints{};
            bestSplitPoints.fill(std::make_pair(0, std::numeric_limits<size_t>::max()));

            MyRansCodedDataWithSplits<NSplits> result{
                    encoder.symbolBuffer.size(), std::move(encoder.bitstream), std::move(encoder.rans),
                    HeuristicSymbolCount, {}
            };

            auto targetSymbolCountPerSplit = saveDiv(encoder.symbolBuffer.size(), NSplits);

            auto stateFrontIt = encoder.intermediateStates.begin();
            auto stateRearIt = encoder.intermediateStates.begin();
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

                    for (auto splitId = 1; splitId < NSplits; splitId++) {
                        int64_t targetSymbolId = targetSymbolCountPerSplit * splitId;
                        auto heuristic = std::abs(static_cast<int64_t>(stateRearIt->symbolId) - targetSymbolId) + std::abs(static_cast<int64_t>(stateRearIt->symbolId + splitZoneSymbolCount) - targetSymbolId);
                        if (heuristic < bestSplitPoints[splitId].second)
                            bestSplitPoints[splitId] = std::make_pair(cutPosition, heuristic);
                    }

                }
            }

            std::array<size_t, NSplits> splitPoints;
            std::transform(bestSplitPoints.begin(), bestSplitPoints.end(), splitPoints.begin(), [](auto v) { return v.first; });
            buildSplitsMetadata(splitPoints, result);

            return result;
        }

        template<size_t NSplits>
        MyRansCodedDataWithSplits<NSplits> flushSplits_equalBitstreamLength() {
            MyRansCodedDataWithSplits<NSplits> result{
                    encoder.symbolBuffer.size(), std::move(encoder.bitstream), std::move(encoder.rans),
                    EqualBitstreamLength, {}
            };

            auto targetLengthPerSplit = saveDiv(result.bitstream.size(), NSplits);
            std::array<size_t, NSplits> splitPoints;
            for (auto splitId = 1; splitId < NSplits; splitId++) {
                splitPoints[splitId] = (NSplits - splitId) * targetLengthPerSplit;
            }
            buildSplitsMetadata(splitPoints, result);

            return result;
        }

        template<size_t NSplits>
        void buildSplitsMetadata(const std::array<size_t, NSplits>& splitPoints, MyRansCodedDataWithSplits<NSplits>& result) {
            result.splits[0] = {
                    result.bitstream.size() - 1,
                    result.finalRans,
                    {}
            };
            for (auto splitId = 1; splitId < NSplits; splitId++) {
                auto splitPoint = splitPoints[splitId];
                auto splitPointIt = encoder.intermediateStates.begin() + splitPoint;
                result.splits[splitId].cutPosition = splitPoint;

                size_t foundEncoderCount = 0;
                while (foundEncoderCount != NInterleaved) {
                    if (!result.splits[splitId].startSymbolGroupIds[splitPointIt->encoderId()]) {
                        result.splits[splitId].intermediateRans[splitPointIt->encoderId()].state = splitPointIt->intermediateState;
                        result.splits[splitId].startSymbolGroupIds[splitPointIt->encoderId()] = splitPointIt->symbolGroupId();
                        foundEncoderCount++;
                    }
                    splitPointIt--;
                }
            }
        }


    };
}

#endif //RECOIL_RANS_SPLIT_ENCODER_H
