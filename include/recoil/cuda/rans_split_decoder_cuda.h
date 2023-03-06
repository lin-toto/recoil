#ifndef RECOIL_RANS_SPLIT_DECODER_CUDA_H
#define RECOIL_RANS_SPLIT_DECODER_CUDA_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/cuda/rans_decoder_cuda.h"

namespace Recoil {
    /*
     * CUDA interface class between host and device code.
     * From this point the host containers will be converted to CUDA equivalents.
     */
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t NInterleaved, size_t NSplits>
    class RansSplitDecoderCuda {
        using MyRansCodedDataWithSplits = RansCodedDataWithSplits<
                RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved, NSplits>;
    public:
        explicit RansSplitDecoderCuda(MyRansCodedDataWithSplits data) {

        }
    protected:

    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_CUDA_H
