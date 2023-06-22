#ifndef RECOIL_RANS_SYMBOL_SPLIT_DECODER_H
#define RECOIL_RANS_SYMBOL_SPLIT_DECODER_H

#include "recoil/lib/math.h"
#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x32.h"
#include <span>
#include <numeric>

#ifdef AVX512
#include "recoil/simd/rans_decoder_avx512_32x32.h"
#endif

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
#ifdef AVX512
        using MyRansDecoder = RansDecoder_AVX512_32x32<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
#else
        using MyRansDecoder = RansDecoder_AVX2_32x32<ValueType, ProbBits, RenormLowerBound, LutGranularity>;
#endif
    public:
        std::vector<ValueType> result;

        explicit RansSymbolSplitDecoder(std::vector<MyRansCodedData> data, const MyCdfLutPool &pool)
            : data(std::move(data)), pool(pool),
              result(std::accumulate(data.begin(), data.end(), 0, [](size_t len, auto &d) { return len + d.symbolCount; })) {}

        void decodeSplit(const size_t splitId, const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            const auto totalSymbolCount = result.size();
            if (splitId >= data.size()) [[unlikely]] throw std::runtime_error("Invalid splitId");

            MyRansDecoder decoder(data[splitId].getRealBitstream(), data[splitId].finalRans, pool);
            auto outOffset = splitId * saveDiv<size_t>(totalSymbolCount, data.size());
            decoder.decode(
                    cdfOffset, lutOffset, data[splitId].symbolCount,
                    (std::span{result}).subspan(outOffset));
        }

        void decodeSplit(size_t splitId, const std::span<CdfLutOffsetType> splitCdfOffsets, const std::span<CdfLutOffsetType> splitLutOffsets) {
            if (splitId >= data.size()) [[unlikely]] throw std::runtime_error("Invalid splitId");

            MyRansDecoder decoder(data[splitId].getRealBitstream(), data[splitId].finalRans, pool);
            auto offset = splitId * saveDiv<size_t>(result.size(), data.size());
            decoder.decode(splitCdfOffsets, splitLutOffsets, (std::span{result}).subspan(offset));
        }
    protected:
        std::vector<MyRansCodedData> data;
        const MyCdfLutPool &pool;
    };
}

#endif //RECOIL_RANS_SYMBOL_SPLIT_DECODER_H
