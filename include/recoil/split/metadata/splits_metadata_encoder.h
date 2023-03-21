#ifndef RECOIL_SPLITS_METADATA_ENCODER_H
#define RECOIL_SPLITS_METADATA_ENCODER_H

#include "recoil/lib/math.h"
#include "recoil/lib/bits_writer.h"
#include "recoil/rans_coded_data.h"

#include <algorithm>
#include <vector>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, size_t NInterleaved>
    class SplitsMetadataEncoder {
        static_assert(1 << (sizeof(RansBitstreamType) * 8) >= RenormLowerBound, "RansBitstreamType is too small to fit encoder states");

        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyRansSplitsMetadata = RansSplitsMetadata<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        explicit SplitsMetadataEncoder(MyRansCodedData data) : data(std::move(data)), metadata(generateDefaultMetadata()) {}
        SplitsMetadataEncoder(MyRansCodedData data, MyRansSplitsMetadata metadata) : data(std::move(data)), metadata(std::move(metadata)) {}

        std::vector<RansBitstreamType> combine() {
            std::vector<RansBitstreamType> combinedBitstream;
            const auto nSplits = metadata.splits.size();

            writer.template write<uint16_t>(nSplits);
            writer.template write<uint32_t>(data.symbolCount);

            if (nSplits > 1) {
                std::vector<int> diffCutPositions, diffSymbolGroupIds;
                for (auto splitId = 1; splitId < nSplits; splitId++) {
                    /*
                     * We expect the bitstream to have a uniform distribution, so distribution of cut positions and
                     * symbol ids should be even. We only code their differences to the estimation to save bit-rate.
                     */
                    auto &split = metadata.splits[splitId];
                    diffCutPositions.push_back(
                            split.cutPosition - (nSplits - splitId) * saveDiv(data.getRealBitstream().size(), nSplits));
                    diffSymbolGroupIds.push_back(split.minSymbolGroupId() - splitId * saveDiv(saveDiv(data.symbolCount,NInterleaved), nSplits));
                }
                auto cutPositionsLength = getMaxActualLength(diffCutPositions);
                auto symbolGroupIdsLength = getMaxActualLength(diffSymbolGroupIds);

                writer.template writeLength<int>(cutPositionsLength);
                for (auto diffCutPosition: diffCutPositions) writer.writeData(diffCutPosition, cutPositionsLength);
                writer.template writeLength<int>(symbolGroupIdsLength);
                for (auto diffSymbolGroupId: diffSymbolGroupIds)
                    writer.writeData(diffSymbolGroupId, symbolGroupIdsLength);

                for (auto it = metadata.splits.begin() + 1; it != metadata.splits.end(); it++) {
                    /*
                     * Similar approach here. Code all symbol group ids in a split related to the minimum id.
                     */
                    std::vector<uint16_t> diffSymbolGroupIdsInSplit;
                    for (auto startSymbolGroupId: it->startSymbolGroupIds) {
                        diffSymbolGroupIdsInSplit.push_back(startSymbolGroupId - it->minSymbolGroupId());
                    }

                    auto symbolGroupIdsInSplitLength = getMaxActualLength(diffSymbolGroupIdsInSplit);
                    writer.template writeLength<uint16_t>(symbolGroupIdsInSplitLength);
                    for (auto diffSymbolGroupIdInSplit: diffSymbolGroupIdsInSplit)
                        writer.writeData(diffSymbolGroupIdInSplit, symbolGroupIdsInSplitLength);
                }
            }

            for (auto rans : data.finalRans) {
                writer.writeData(rans.state, sizeof(RansStateType) * 8);
            }

            std::copy(writer.buf.begin(), writer.buf.end(), std::back_inserter(combinedBitstream));
            writer.reset();

            for (auto it = metadata.splits.begin() + 1; it != metadata.splits.end(); it++) {
                for (auto rans : it->intermediateRans)
                    combinedBitstream.push_back(rans.state);
            }

            std::copy(data.getRealBitstream().begin(), data.getRealBitstream().end(), std::back_inserter(combinedBitstream));
            return combinedBitstream;
        }
    protected:
        MyRansCodedData data;
        MyRansSplitsMetadata metadata;

        BitsWriter<RansBitstreamType> writer;

        template<typename T>
        uint8_t getMaxActualLength(const std::vector<T> &arr) const {
            return writer.getActualLength(*std::max_element(
                    arr.begin(), arr.end(), [](const auto& a, const auto& b) { return abs(a) < abs(b); }));
        }

        MyRansSplitsMetadata generateDefaultMetadata() {
            MyRansSplitsMetadata metadata{ EqualBitstreamLength, {} };
            metadata.splits.resize(1);
            metadata.splits[0] = { data.getRealBitstream().size() - 1, data.finalRans, {} };
            return metadata;
        }
    };
}

#endif //RECOIL_SPLITS_METADATA_ENCODER_H

