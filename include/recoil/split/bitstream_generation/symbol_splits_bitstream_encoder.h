#ifndef RECOIL_SYMBOL_SPLITS_BITSTREAM_ENCODER_H
#define RECOIL_SYMBOL_SPLITS_BITSTREAM_ENCODER_H

#include "recoil/lib/math.h"
#include "recoil/lib/bits_writer.h"
#include "recoil/rans_coded_data.h"

#include <algorithm>
#include <vector>
#include <numeric>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, size_t NInterleaved>
    class SymbolSplitsBitstreamEncoder {
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        explicit SymbolSplitsBitstreamEncoder(std::vector<MyRansCodedData> data) : data(std::move(data)) {}

        std::vector<RansBitstreamType> combine() {
            std::vector<RansBitstreamType> combinedBitstream;
            const auto nSplits = data.size();
            writer.template write<uint16_t>(nSplits);

            auto totalSymbolCount = std::accumulate(data.begin(), data.end(), 0, [](size_t len, auto &v) { return len + v.symbolCount; });
            writer.template write<uint32_t>(totalSymbolCount);

            auto totalRawBitstreamLength = std::accumulate(
                    data.begin(), data.end(), 0, [](size_t len, auto &d) { return len + d.getRealBitstream().size(); });
            std::vector<int32_t> bitstreamLengthDiffs(data.size());
            std::transform(data.begin(), data.end(), bitstreamLengthDiffs.begin(), [totalRawBitstreamLength, nSplits] (auto &d) {
                return static_cast<int32_t>(d.getRealBitstream().size()) - saveDiv<size_t>(totalRawBitstreamLength, nSplits);
            });

            auto bitstreamLengthDiffsLength = writer.getMaxActualLength(bitstreamLengthDiffs);
            writer.template writeLength<int32_t>(bitstreamLengthDiffsLength);
            for (auto bitstreamLengthDiff : bitstreamLengthDiffs)
                writer.template writeData<int32_t>(bitstreamLengthDiff, bitstreamLengthDiffsLength);

            writer.forward();

            for (auto &d : data) {
                for (auto &rans : d.finalRans)
                    writer.writeData(rans.state, sizeof(RansStateType) * 8);
            }

            std::copy(writer.buf.begin(), writer.buf.end(), std::back_inserter(combinedBitstream));
            writer.reset();

            for (auto &d : data)
                std::copy(d.getRealBitstream().begin(), d.getRealBitstream().end(), std::back_inserter(combinedBitstream));

            return combinedBitstream;
        }
    protected:
        std::vector<MyRansCodedData> data;

        BitsWriter<RansBitstreamType> writer;
    };
}
#endif //RECOIL_SYMBOL_SPLITS_BITSTREAM_ENCODER_H
