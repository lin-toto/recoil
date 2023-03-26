#ifndef RECOIL_SYMBOL_SPLITS_BITSTREAM_DECODER_H
#define RECOIL_SYMBOL_SPLITS_BITSTREAM_DECODER_H

#include "recoil/lib/bits_reader.h"
#include "recoil/rans_coded_data.h"

#include <vector>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, size_t NInterleaved>
    class SymbolSplitsBitstreamDecoder {
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        explicit SymbolSplitsBitstreamDecoder(std::vector<RansBitstreamType> bitstream)
            : bitstream(std::move(bitstream)), reader(std::span{this->bitstream}) {}

        std::vector<MyRansCodedData> decode() {
            auto nSplits = reader.template read<uint16_t>();
            auto totalSymbolCount = reader.template read<uint32_t>();

            std::vector<MyRansCodedData> data;

            std::vector<int32_t> bitstreamLengthDiffs(nSplits);
            auto bitstreamLengthDiffsLength = reader.template readLength<int32_t>();
            for (auto &bitstreamLengthDiff : bitstreamLengthDiffs)
                bitstreamLengthDiff = reader.template readData<int32_t>(bitstreamLengthDiffsLength);

            reader.forward();
            for (auto &d : data) {
                for (auto &rans : d.finalRans)
                    rans.state = reader.template readData<RansBitstreamType>(sizeof(RansStateType) * 8);
            }

            auto it = bitstream.begin() + reader.currentIteratorPosition() + 1;
            auto bitstreamLengthSum = bitstream.end() - it;
            for (int splitId = 0; splitId < nSplits; splitId++) {
                const auto symbolsPerSplit = saveDiv<size_t>(totalSymbolCount, nSplits);
                data[splitId].symbolCount = splitId == nSplits - 1 ? totalSymbolCount % symbolsPerSplit : symbolsPerSplit;

                data[splitId].bitstream.resize(16);
                data[splitId].leftPadding = 16;

                auto bitstreamLength = saveDiv(bitstreamLengthSum, nSplits) + bitstreamLengthDiffs[splitId];
                std::copy(it,
                          it + bitstreamLength,
                          std::back_inserter(data[splitId].bitstream));
                it += bitstreamLength;
            }

            return data;
        }
    protected:
        std::vector<RansBitstreamType> bitstream;
        BitsReader<RansBitstreamType> reader;
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, size_t NInterleaved>
    using SymbolSplitsBitstreamDecoder_Rans64 = SymbolSplitsBitstreamDecoder<uint16_t, ValueType, uint64_t, uint32_t, ProbBits, 1ull << 31, 32, NInterleaved>;

    template<std::unsigned_integral ValueType, uint8_t ProbBits, size_t NInterleaved>
    using SymbolSplitsBitstreamDecoder_Rans32 = SymbolSplitsBitstreamDecoder<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, 1u << 16, 16, NInterleaved>;
}

#endif //RECOIL_SYMBOL_SPLITS_BITSTREAM_DECODER_H
