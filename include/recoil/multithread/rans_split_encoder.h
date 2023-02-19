#ifndef RECOIL_RANS_SPLIT_ENCODER_H
#define RECOIL_RANS_SPLIT_ENCODER_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_encoder.h"
#include <vector>
#include <numeric>

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

        template<size_t NSplits>
        MyRansCodedDataWithSplits<NSplits> flushSplits() {
            encoder.encodeAll();

            std::vector<size_t> splitZoneSymbolCounts;
            splitZoneSymbolCounts.reserve(encoder.bitstream.size());

            {
                // Step 1: scan bitstream for all potential split zones
                std::array<size_t, NInterleaved> splitZoneEncoderCount{};

                auto stateFrontIt = encoder.intermediateStates.begin();
                auto stateRearIt = encoder.intermediateStates.begin();
                auto symbolCountIt = splitZoneSymbolCounts.begin();
                for (bool initial = true;
                     stateRearIt != encoder.intermediateStates.end(); stateRearIt++, symbolCountIt++) {
                    splitZoneEncoderCount[stateRearIt->encoderId()]++;

                    if (initial) {
                        if (std::transform_reduce(
                                splitZoneEncoderCount.begin(), splitZoneEncoderCount.end(), 0, std::plus{},
                                [](auto val) { return val >= 1 ? 1 : 0; }) != NInterleaved) {
                            continue;
                        } else {
                            initial = false;
                        }
                    }

                    while (splitZoneEncoderCount[stateFrontIt->encoderId()] > 1) {
                        splitZoneEncoderCount[stateFrontIt->encoderId()]--;
                        stateFrontIt++;
                    }

                    *symbolCountIt = NInterleaved * (stateRearIt->symbolGroupId() - stateFrontIt->symbolGroupId() + 1);
                }
            }

            {
                // Step 2: scan bitstream again to split it as even as possible
                // TODO: maybe need a better strategy?


            }


        }
    protected:
        MyRansEncoder encoder;
    };
}

#endif //RECOIL_RANS_SPLIT_ENCODER_H
