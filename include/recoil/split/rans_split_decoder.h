#ifndef RECOIL_RANS_SPLIT_DECODER_H
#define RECOIL_RANS_SPLIT_DECODER_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"
#include "recoil/simd/rans_decoder_avx2_32x32.h"
#include <span>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved>
    class RansSplitDecoder {
    protected:
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyRansSplitsMetadata = RansSplitsMetadata<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;

        // TODO: allow any class derived from RansDecoder, from a template parameter
        //using MyRansDecoder = RansDecoder<
        //        CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved>;
        //using MyRansDecoder = RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, NInterleaved>;
        using MyRansDecoder = RansDecoder_AVX2_32x32<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
    public:
        std::vector<ValueType> result;

        explicit RansSplitDecoder(MyRansCodedData data, MyRansSplitsMetadata metadata, const MyCdfLutPool &pool)
            : data(std::move(data)), metadata(std::move(metadata)), pool(pool), result(data.symbolCount) {}

        void decodeSplit(const size_t splitId, const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            if (splitId >= metadata.splits.size()) [[unlikely]] throw std::runtime_error("Invalid splitId");

            auto& currentSplit = metadata.splits[splitId];
            MyRansDecoder decoder(
                    std::span(data.getRealBitstream().data(), currentSplit.cutPosition + 1),
                    currentSplit.intermediateRans, pool);

            if (splitId != 0) {
                std::array<CdfLutOffsetType, NInterleaved> cdfOffsets, lutOffsets;
                cdfOffsets.fill(cdfOffset);
                lutOffsets.fill(lutOffset);

                std::array<bool, NInterleaved> ransInitializedState{};

                bool ransAllInitialized = false;
                for (size_t symbolGroupId = currentSplit.minSymbolGroupId(); !ransAllInitialized; symbolGroupId++) {
                    ransAllInitialized = syncRansOnce(decoder, currentSplit, symbolGroupId, ransInitializedState, cdfOffsets, lutOffsets);
                }
            }

            std::span resultSpan{result};
            size_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId() + 1);
            size_t decodeEndSymbolId = splitId == metadata.splits.size() - 1 ? data.symbolCount
                    : NInterleaved * (1 + metadata.splits[splitId + 1].maxSymbolGroupId());
            decoder.decode(cdfOffset, lutOffset, decodeEndSymbolId - decodeStartSymbolId, resultSpan.subspan(decodeStartSymbolId));
        }

        void decodeSplit(size_t splitId, const std::span<CdfLutOffsetType> allCdfOffsets, const std::span<CdfLutOffsetType> allLutOffsets) {
            if (splitId >= metadata.splits.size()) [[unlikely]] throw std::runtime_error("Invalid splitId");
            if (allCdfOffsets.size() != allLutOffsets.size()) [[unlikely]] throw std::runtime_error("CDF and LUT offset length mismatch");
            if (allCdfOffsets.size() != data.symbolCount) [[unlikely]] throw std::runtime_error("Need the full CDF");

            auto& currentSplit = metadata.splits[splitId];
            MyRansDecoder decoder(
                    std::span(data.getRealBitstream().data(), currentSplit.cutPosition + 1),
                    currentSplit.intermediateRans, pool);

            if (splitId != 0) {
                std::array<CdfLutOffsetType, NInterleaved> cdfOffsets, lutOffsets;
                std::array<bool, NInterleaved> ransInitializedState{};

                bool ransAllInitialized = false;
                for (size_t symbolGroupId = currentSplit.minSymbolGroupId(); !ransAllInitialized; symbolGroupId++) {
                    size_t cdfLutOffset = NInterleaved * (symbolGroupId - currentSplit.minSymbolGroupId());
                    ransAllInitialized = syncRansOnce(decoder, currentSplit, symbolGroupId, ransInitializedState,
                                                      allCdfOffsets.subspan(cdfLutOffset),
                                                      allLutOffsets.subspan(cdfLutOffset));
                }
            }

            std::span resultSpan{result};
            size_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId() + 1);
            size_t decodeEndSymbolId = splitId == metadata.splits.size() - 1 ? data.symbolCount
                                                                             : NInterleaved * (1 + metadata.splits[splitId + 1].maxSymbolGroupId());
            size_t cdfLutOffset = NInterleaved * (currentSplit.maxSymbolGroupId() + 1 - currentSplit.minSymbolGroupId());
            decoder.decode(allCdfOffsets.subspan(cdfLutOffset), allLutOffsets.subspan(cdfLutOffset),
                           decodeEndSymbolId - decodeStartSymbolId, resultSpan.subspan(decodeStartSymbolId));
        }
    protected:
        MyRansCodedData data;
        MyRansSplitsMetadata metadata;
        const MyCdfLutPool &pool;

        inline bool syncRansOnce(MyRansDecoder& decoder, const MyRansSplitsMetadata::Split& currentSplit,
                                 const size_t symbolGroupId, std::array<bool, NInterleaved>& ransInitializedState,
                                 const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets) {
            bool ransAllInitialized = true;
            for (size_t decoderId = 0; decoderId < NInterleaved; decoderId++) {
                if (!ransInitializedState[decoderId]) {
                    if (currentSplit.startSymbolGroupIds[decoderId] == symbolGroupId) {
                        decoder.renorm(decoder.rans[decoderId]);
                        ransInitializedState[decoderId] = true;
                    } else ransAllInitialized = false;
                } else decoder.decodeSymbol(decoder.rans[decoderId], cdfOffsets[decoderId], lutOffsets[decoderId]);
            }

            return ransAllInitialized;
        }
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_H
