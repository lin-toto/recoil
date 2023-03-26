#ifndef RECOIL_RANS_SYMBOL_SPLIT_ENCODER_H
#define RECOIL_RANS_SYMBOL_SPLIT_ENCODER_H

#include "recoil/lib/math.h"
#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_encoder.h"
#include <vector>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved>
    class RansSymbolSplitEncoder {
        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        // TODO: allow any class derived from RansEncoder, from a template parameter (we may have an AVX encoder in the future)
        using MyRansEncoder = RansEncoder<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved, true>;
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        RansSymbolSplitEncoder(std::array<MyRans, NInterleaved> rans, const MyCdfLutPool& pool)
            : dummyEncoder(std::move(rans), pool), pool(pool) {}

        inline MyRansEncoder& getEncoder() { return dummyEncoder; }

        std::vector<MyRansCodedData> flushSplits(size_t nSplits) {
            std::vector<MyRansCodedData> results;

            auto symbolsPerSplit = saveDiv<size_t>(dummyEncoder.symbolBuffer.size(), nSplits);
            for (int splitId = 0; splitId < nSplits; splitId++) {
                MyRansEncoder enc(dummyEncoder.rans, pool);
                auto symbols = (std::span{dummyEncoder.symbolBuffer}).subspan(
                        symbolsPerSplit * splitId,
                        splitId == nSplits - 1 ? std::dynamic_extent : symbolsPerSplit);

                std::copy(symbols.begin(), symbols.end(), std::back_inserter(enc.symbolBuffer));

                results.push_back(enc.flush());
            }

            return results;
        }

    protected:
        /*
         * We use a dummy encoder to do symbol buffering.
         * Then, we steal its buffered symbols and encode with multiple encoders.
         */
        MyRansEncoder dummyEncoder;
        const MyCdfLutPool& pool;
    };
}

#endif //RECOIL_RANS_SYMBOL_SPLIT_ENCODER_H
