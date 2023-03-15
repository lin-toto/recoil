#ifndef RECOIL_RANS_CODED_DATA_H
#define RECOIL_RANS_CODED_DATA_H

#include <vector>
#include <concepts>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, size_t NInterleaved>
    struct RansCodedData {
        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        size_t symbolCount;
        std::vector<RansBitstreamType> bitstream;
        std::array<MyRans, NInterleaved> finalRans;
    };

    enum SplitStrategy {
        HeuristicSymbolCount, EqualBitstreamLength
    };

    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, size_t NInterleaved>
    struct RansSplitsMetadata {
        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;

        SplitStrategy splitStrategy;

        struct Split {
            size_t cutPosition; // TODO: need to verify this is actually safe!
            std::array<MyRans, NInterleaved> intermediateRans;
            std::array<size_t, NInterleaved> startSymbolGroupIds;

            [[nodiscard]] inline size_t minSymbolGroupId() const { return *std::min_element(startSymbolGroupIds.begin(), startSymbolGroupIds.end()); }
            [[nodiscard]] inline size_t maxSymbolGroupId() const { return *std::max_element(startSymbolGroupIds.begin(), startSymbolGroupIds.end()); }
        };

        /*
         * For split 0, it always starts at position 0; intermediateRans == finalRans[0];
         * and startSymbolGroupIds is 0.
         */
        std::vector<Split> splits;
    };
}

#endif //RECOIL_RANS_CODED_DATA_H
