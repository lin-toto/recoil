#ifndef RECOIL_RANS_SPLIT_ENCODER_H
#define RECOIL_RANS_SPLIT_ENCODER_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/rans_encoder.h"

namespace Recoil {
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t NInterleaved, size_t NSplits>
    class RansSplitEncoder {

    };
}

#endif //RECOIL_RANS_SPLIT_ENCODER_H
