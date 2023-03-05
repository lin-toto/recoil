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
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
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

        enum SplitStrategy {
            HeuristicSymbolCount, EqualLength
        };

        template<size_t NSplits>
        MyRansCodedDataWithSplits<NSplits> flushSplits(SplitStrategy strategy = HeuristicSymbolCount) {
            encoder.encodeAll();

            switch (strategy) {
                case HeuristicSymbolCount:
                    return flushSplits_heuristicSymbolCount<NSplits>();
                case EqualLength:
                    // TODO
                    break;
            }
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
                    encoder.symbolBuffer.size(), std::move(encoder.bitstream), std::move(encoder.rans), {}
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

            result.splits[0] = {
                result.bitstream.size(),
                result.finalRans,
                {}
            };
            for (auto splitId = 1; splitId < NSplits; splitId++) {
                auto splitPoint = bestSplitPoints[splitId].first;
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

            //encoder.reset();

            return result;
        }
    };
}

#endif //RECOIL_RANS_SPLIT_ENCODER_H
