#ifndef RECOIL_SPLITS_METADATA_DECODER_H
#define RECOIL_SPLITS_METADATA_DECODER_H

#include "recoil/lib/bits_reader.h"
#include "recoil/rans_coded_data.h"

#include <vector>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, size_t NInterleaved>
    class SplitsMetadataDecoder {
        static_assert(1 << (sizeof(RansBitstreamType) * 8) >= RenormLowerBound, "RansBitstreamType is too small to fit encoder states");

        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyRansSplitsMetadata = RansSplitsMetadata<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        explicit SplitsMetadataDecoder(std::vector<RansBitstreamType> bitstream)
            : bitstream(std::move(bitstream)), reader(std::span{this->bitstream}) {}

        std::pair<MyRansCodedData, MyRansSplitsMetadata> decode() {
            MyRansCodedData data;
            MyRansSplitsMetadata metadata;
            std::vector<int> diffCutPositions(1), minSymbolGroupIds(1);

            for (auto &rans : data.finalRans)
                rans.state = reader.template readData<RansStateType>(sizeof(RansStateType) * 8);

            size_t nSplits = reader.template read<uint16_t>();
            data.symbolCount = reader.template read<uint32_t>();

            metadata.splits.resize(nSplits);
            metadata.splits[0].startSymbolGroupIds.fill(0);
            metadata.splits[0].intermediateRans = data.finalRans;

            if (nSplits > 1) {
                auto cutPositionsLength = reader.template readLength<int>();
                for (auto splitId = 1; splitId < nSplits; splitId++)
                    diffCutPositions.push_back(reader.template readData<int>(cutPositionsLength));
                auto symbolGroupIdsLength = reader.template readLength<int>();
                for (auto splitId = 1; splitId < nSplits; splitId++)
                    minSymbolGroupIds.push_back(reader.template readData<int>(symbolGroupIdsLength) +
                                                splitId * saveDiv(saveDiv(data.symbolCount, NInterleaved), nSplits));

                for (auto splitId = 1; splitId < nSplits; splitId++) {
                    auto symbolGroupIdsInSplitLength = reader.template readLength<uint16_t>();
                    for (auto &startSymbolGroupId: metadata.splits[splitId].startSymbolGroupIds)
                        startSymbolGroupId = reader.template readData<uint16_t>(symbolGroupIdsInSplitLength) +
                                             minSymbolGroupIds[splitId];
                }
            }

            auto it = bitstream.begin() + reader.currentIteratorPosition() + 1;

            for (auto splitId = 1; splitId < nSplits; splitId++) {
                auto &split = metadata.splits[splitId];
                for (auto &rans : split.intermediateRans) {
                    rans.state = *it;
                    it++;
                }
            }

            data.leftPadding = it - bitstream.begin();
            data.bitstream = std::move(bitstream);
            if (data.leftPadding < 16) {
                // Pad the left-side of bitstream with 16-bytes per AVX-512 decoder requirement.
                data.bitstream.insert(data.bitstream.begin(), 16 - data.leftPadding, 0x00);
                data.leftPadding = 16;
            }

            metadata.splits[0].cutPosition = data.getRealBitstream().size() - 1;
            for (auto splitId = 1; splitId < nSplits; splitId++) {
                metadata.splits[splitId].cutPosition = diffCutPositions[splitId] + (nSplits - splitId) * saveDiv(data.getRealBitstream().size(), nSplits);
            }

            return std::make_pair(std::move(data), std::move(metadata));
        }
    protected:
        std::vector<RansBitstreamType> bitstream;
        BitsReader<RansBitstreamType> reader;
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, size_t NInterleaved>
    using SplitsMetadataDecoder_Rans64 = SplitsMetadataDecoder<uint16_t, ValueType, uint64_t, uint32_t, ProbBits, 1ull << 31, 32, NInterleaved>;

    template<std::unsigned_integral ValueType, uint8_t ProbBits, size_t NInterleaved>
    using SplitsMetadataDecoder_Rans32 = SplitsMetadataDecoder<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, 1u << 16, 16, NInterleaved>;
}

#endif //RECOIL_SPLITS_METADATA_DECODER_H
