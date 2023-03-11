#ifndef RECOIL_RANS_SPLIT_DECODER_H
#define RECOIL_RANS_SPLIT_DECODER_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"
#include <span>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved, size_t NSplits>
    class RansSplitDecoder {
    protected:
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MyRansCodedDataWithSplits = RansCodedDataWithSplits<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved, NSplits>;

        // TODO: allow any class derived from RansDecoder, from a template parameter
        //using MyRansDecoder = RansDecoder<
        //        CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved>;
        using MyRansDecoder = RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, NInterleaved>;
    public:
        explicit RansSplitDecoder(MyRansCodedDataWithSplits data, const MyCdfLutPool& pool) : data(std::move(data)), pool(pool) {}

        std::vector<ValueType> decodeSplit(const size_t splitId, const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            auto& currentSplit = data.splits[splitId];
            MyRansDecoder decoder(
                    std::span(data.bitstream.data(), currentSplit.cutPosition + 1),
                    currentSplit.intermediateRans, pool);

            if (splitId != 0) {
                // synchronize decoders
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
                        } else decoder.decodeSymbol(decoder.rans[decoderId], cdfOffset, lutOffset);
                    }
                }
            }

            size_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId() + 1);
            size_t decodeEndSymbolId = splitId == NSplits - 1 ? data.symbolCount
                    : NInterleaved * (1 + data.splits[splitId + 1].maxSymbolGroupId());
            return decoder.decode(cdfOffset, lutOffset, decodeEndSymbolId - decodeStartSymbolId);
        }

        std::vector<ValueType> decodeSplit(size_t splitId, const std::span<CdfLutOffsetType> allCdfOffsets, const std::span<CdfLutOffsetType> allLutOffsets) {
            // TODO
        }
    protected:
        MyRansCodedDataWithSplits data;
        const MyCdfLutPool &pool;
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_H
