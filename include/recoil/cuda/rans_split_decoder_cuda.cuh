#ifndef RECOIL_RANS_SPLIT_DECODER_CUDA_H
#define RECOIL_RANS_SPLIT_DECODER_CUDA_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/cuda/rans_decoder_cuda.h"
#include "recoil/cuda/macros.h"

#include <cuda_runtime_api.h>

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
        explicit RansSplitDecoderCuda(const MyRansCodedDataWithSplits& data) : symbolCount(data.symbolCount) {
            cudaMalloc(&bitstream, sizeof(RansBitstreamType) * data.bitstream.size());
            cudaMemcpy(bitstream, data.bitstream.data(), sizeof(RansBitstreamType) * data.bitstream.size(), cudaMemcpyHostToDevice);
        }

        ~RansSplitDecoderCuda() {
            cudaFree(bitstream);
        }

        // TODO: proper CDF store

        std::vector<RansBitstreamType> decodeAll() {
            CUDA_DEVICE_PTR ValueType *outputBuffer;
            cudaMalloc(&outputBuffer, sizeof(ValueType) * symbolCount);

            decodeKernel<<<NSplits, NInterleaved>>>();

            cudaFree(outputBuffer);
        }

    protected:
        size_t symbolCount;

        CUDA_DEVICE_PTR RansBitstreamType *bitstream;

        CUDA_GLOBAL void decodeKernel() {

        }
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_CUDA_H
