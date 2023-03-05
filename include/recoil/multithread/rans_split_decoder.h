#ifndef RECOIL_RANS_SPLIT_DECODER_H
#define RECOIL_RANS_SPLIT_DECODER_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_decoder.h"
#include "recoil/lib/cdf.h"
#include <span>

namespace Recoil {
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t NInterleaved, size_t NSplits>
    class RansSplitDecoder {
        // TODO: allow any class derived from RansDecoder, from a template parameter
    protected:
        using MyRansCodedDataWithSplits = RansCodedDataWithSplits<
                RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved, NSplits>;
        using MyRansDecoder = RansDecoder<
                RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        explicit RansSplitDecoder(MyRansCodedDataWithSplits data) : data(std::move(data)) {}

        std::vector<ValueType> decodeSplit(const size_t splitId, const Cdf cdf) {
            auto& currentSplit = data.splits[splitId];
            MyRansDecoder decoder(
                    std::span(data.bitstream.data(), std::min(data.bitstream.size(), currentSplit.cutPosition + 1)),
                    currentSplit.intermediateRans);

            if (splitId != 0) {
                // Step 1: synchronize decoders
                std::array<bool, NInterleaved> ransInitialized{};
                bool ransAllInitialized = false;

                for (size_t symbolGroupId = currentSplit.minSymbolGroupId(); !ransAllInitialized; symbolGroupId++) {
                    ransAllInitialized = true;
                    for (size_t decoderId = 0; decoderId < NInterleaved; decoderId++) {
                        if (!ransInitialized[decoderId]) {
                            if (currentSplit.startSymbolGroupIds[decoderId] == symbolGroupId) {
                                decoder.renorm(decoder.rans[decoderId]);
                                ransInitialized[decoderId] = true;
                            } else ransAllInitialized = false;
                        } else decoder.decodeSymbol(decoder.rans[decoderId], cdf);
                    }
                }
            }

            size_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId() + 1);
            size_t decodeEndSymbolId = splitId == NSplits - 1 ? data.symbolCount
                    : NInterleaved * (1 + data.splits[splitId + 1].maxSymbolGroupId());
            return decoder.decode(cdf, decodeEndSymbolId - decodeStartSymbolId);
        }

        std::vector<ValueType> decodeSplit(size_t splitId, const std::span<Cdf> fullCdf) {
            // TODO
        }
    protected:
        MyRansCodedDataWithSplits data;
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_H
