#ifndef RECOIL_RANS_CODED_DATA_H
#define RECOIL_RANS_CODED_DATA_H

#include <vector>
#include <array>

namespace Recoil {
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t nInterleaved>
    struct RansCodedData {
        using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        std::vector<RansBitstreamType> bitstream;
        std::array<MyRans, nInterleaved> finalRans;
    };
}

#endif //RECOIL_RANS_CODED_DATA_H
