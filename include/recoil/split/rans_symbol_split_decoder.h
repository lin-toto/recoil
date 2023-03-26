#ifndef RECOIL_RANS_SYMBOL_SPLIT_DECODER_H
#define RECOIL_RANS_SYMBOL_SPLIT_DECODER_H

#include "recoil/lib/math.h"
#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"
#include "recoil/simd/rans_decoder_avx2_32x32.h"
#include <span>
#include <numeric>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved>
    class RansSymbolSplitDecoder {
    protected:
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;

        // TODO: allow any class derived from RansDecoder, from a template parameter
        //using MyRansDecoder = RansDecoder<
        //        CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved>;
        //using MyRansDecoder = RansDecoder_AVX2_32x8n<ValueType, ProbBits, RenormLowerBound, LutGranularity, NInterleaved>;
        using MyRansDecoder = RansDecoder_AVX2_32x32<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
    public:
        std::vector<ValueType> result;

        explicit RansSymbolSplitDecoder(std::vector<MyRansCodedData> data, const MyCdfLutPool &pool)
            : data(std::move(data)), pool(pool),
              totalSymbolCount(std::accumulate(data.begin(), data.end(), 0, [](auto &v) { return v.symbolCount; })),
              result(totalSymbolCount) {}

        void decodeSplit(const size_t splitId, const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            if (splitId >= data.size()) [[unlikely]] throw std::runtime_error("Invalid splitId");

            MyRansDecoder decoder(data[splitId].getRealBitstream().data(), data[splitId].finalRans, pool);
            auto outOffset = splitId * saveDiv<size_t>(totalSymbolCount, data.size());
            decoder.decode(
                    cdfOffset, lutOffset, data[splitId].symbolCount,
                    (std::span{result}).subspan(outOffset));
        }

        void decodeSplit(size_t splitId, const std::span<CdfLutOffsetType> allCdfOffsets, const std::span<CdfLutOffsetType> allLutOffsets) {
            if (splitId >= data.size()) [[unlikely]] throw std::runtime_error("Invalid splitId");
            if (allCdfOffsets.size() != allLutOffsets.size()) [[unlikely]] throw std::runtime_error("CDF and LUT offset length mismatch");
            if (allCdfOffsets.size() != totalSymbolCount) [[unlikely]] throw std::runtime_error("Need the full CDF");

            MyRansDecoder decoder(data[splitId].getRealBitstream().data(), data[splitId].finalRans, pool);
            auto offset = splitId * saveDiv<size_t>(totalSymbolCount, data.size());
            decoder.decode(
                    allCdfOffsets.subspan(offset), allLutOffsets.subspan(offset),
                    data[splitId].symbolCount, (std::span{result}).subspan(offset));
        }
    protected:
        std::vector<MyRansCodedData> data;
        const MyCdfLutPool &pool;
        size_t totalSymbolCount;
    };
}

#endif //RECOIL_RANS_SYMBOL_SPLIT_DECODER_H
