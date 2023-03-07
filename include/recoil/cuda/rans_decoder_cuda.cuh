#ifndef RECOIL_RANS_DECODER_CUDA_CUH
#define RECOIL_RANS_DECODER_CUDA_CUH

#include "macros.h"
#include "recoil/rans.h"
#include <cuda/std/array>
#include "device_launch_parameters.h"

namespace {
    CUDA_DEVICE inline unsigned getLaneMaskLe() {
        unsigned mask;
        asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
        return mask;
    }
}

namespace Recoil {
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits>
    class RansDecoderCuda {
        const int NInterleaved = 32; // FIXME

        using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
    public:
        CUDA_DEVICE RansDecoderCuda(
                CUDA_DEVICE_PTR RansBitstreamType *bitstream, uint32_t bitstreamOffset,
                CUDA_DEVICE_PTR ValueType *output, uint32_t outputOffset,
                MyRans rans) : bitstreamPtr(bitstream + bitstreamOffset), outputPtr(output + outputOffset), decoder(rans) {}

        CUDA_DEVICE void decode(
                CUDA_DEVICE_PTR const CdfType * __restrict__ cdf,
                CUDA_DEVICE_PTR const ValueType * __restrict__ lut,
                const uint32_t count) {
            const unsigned int decoderId = threadIdx.x;

            for (uint32_t i = 0; i + NInterleaved <= count; i += NInterleaved) {
                outputPtr[decoderId] = decodeOnce(cdf, lut);
                outputPtr += NInterleaved;
            }

            if (decoderId < count % NInterleaved) {
                outputPtr[decoderId] = decodeOnce(cdf, lut);
                outputPtr += count % NInterleaved;
            }
        }

        CUDA_DEVICE inline ValueType decodeOnce(
                CUDA_DEVICE_PTR const CdfType * __restrict__ cdf,
                CUDA_DEVICE_PTR const ValueType * __restrict__ lut) {
            auto probability = decoder.decGetProbability();
            auto symbol = lut[probability];
            //printf("threadid %d sym %c\n", threadIdx.x, symbol);

            // TODO: add LUT + CDF mixed support
            auto start = cdf[symbol];
            auto frequency = cdf[symbol + 1] - cdf[symbol];
            decoder.decAdvanceSymbol(start, frequency);

            renorm();

            return symbol;
        }

        CUDA_DEVICE inline uint8_t renorm() {
            bool shouldRenorm = decoder.decShouldRenorm();
            // TODO: fix ballot_sync flag to support other than 32 threads
            auto vote = __ballot_sync(0xffffffff, shouldRenorm);

            if (shouldRenorm) {
                auto offset = 1 - __popc(vote & getLaneMaskLe());
                decoder.decRenormOnce(bitstreamPtr[offset]);
                //printf("threadid %d renorm read from %d\n", threadIdx.x, offset);
            }

            auto amountRead = __popc(vote);
            bitstreamPtr -= amountRead;

            return amountRead;
        }

    //protected:
        MyRans decoder;
        CUDA_DEVICE_PTR RansBitstreamType * bitstreamPtr;
        CUDA_DEVICE_PTR ValueType * outputPtr;
    };
}

#endif //RECOIL_RANS_DECODER_CUDA_CUH
