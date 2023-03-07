#ifndef RECOIL_RANS_DECODER_CUDA_H
#define RECOIL_RANS_DECODER_CUDA_H

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
        CUDA_DEVICE RansDecoderCuda(RansBitstreamType *bitstream, uint32_t offset, MyRans rans)
                : bitstreamPtr(bitstream + offset), decoder(rans) {
        }

        CUDA_DEVICE void decode(
                CUDA_DEVICE_PTR const CdfType * __restrict__ cdf,
                CUDA_DEVICE_PTR const ValueType * __restrict__ lut,
                const uint32_t count) {
            const unsigned int splitId = blockIdx.x, decoderId = threadIdx.x;
            for (uint32_t i = 0; i + NInterleaved < count; i += NInterleaved) {
                //printf("Iteration: %u\n", i);
                auto amountRead = decodeOnce(cdf, lut);
                bitstreamPtr -= amountRead;
            }

            if (decoderId < count % NInterleaved) {
                //printf("Remaining\n");
                auto amountRead = decodeOnce(cdf, lut);
                bitstreamPtr -= amountRead;
            }
        }
    protected:
        CUDA_DEVICE_PTR RansBitstreamType * __restrict__ bitstreamPtr;
        MyRans decoder;

        CUDA_DEVICE inline uint32_t decodeOnce(
                CUDA_DEVICE_PTR const CdfType * __restrict__ cdf,
                CUDA_DEVICE_PTR const ValueType * __restrict__ lut) {
            const unsigned int splitId = blockIdx.x, decoderId = threadIdx.x;

            auto probability = decoder.decGetProbability();
            auto symbol = lut[probability];
            printf("%c", symbol);
            //printf("idx: %d, state: %x, %c\n", decoderId, decoder.state, symbol);
            // TODO: add LUT + CDF mixed support
            auto start = cdf[symbol];
            auto frequency = cdf[symbol + 1] - cdf[symbol];
            decoder.decAdvanceSymbol(start, frequency);

            bool shouldRenorm = decoder.decShouldRenorm();
            // TODO: fix ballot_sync flag to support other than 32 threads
            auto vote = __ballot_sync(0xffffffff, shouldRenorm);

            if (shouldRenorm) {
                //printf("idx: %d renormalize, lanemask: %x\n", decoderId, getLaneMaskLe());
                auto offset = 1 - __popc(vote & getLaneMaskLe());
                decoder.decRenormOnce(bitstreamPtr[offset]);
            }

            return __popc(vote);
        }
    };

    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t NInterleaved, size_t NSplits>
    RansDecoderCuda(const RansBitstreamType *bitstream,
                    uint32_t offsets,
                    Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits> rans)
    -> RansDecoderCuda<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
}

#endif //RECOIL_RANS_DECODER_CUDA_H
