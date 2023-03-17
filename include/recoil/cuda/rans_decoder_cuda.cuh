#ifndef RECOIL_RANS_DECODER_CUDA_CUH
#define RECOIL_RANS_DECODER_CUDA_CUH

#include "recoil/lib/cuda/macros.h"
#include "recoil/lib/cuda/cuda_libs.cuh"
#include "recoil/rans.h"
#include <cuda/std/array>
#include "device_launch_parameters.h"
#include "rans_coded_data_cuda.h"

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity>
    class RansDecoderCuda {
        static const int NInterleaved = 32;

        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MySymbolLookup = SymbolLookup<CdfType, ValueType, ProbBits, LutGranularity>;
    public:
        CUDA_DEVICE RansDecoderCuda(
                CUDA_DEVICE_PTR const RansBitstreamType *bitstream, uint32_t bitstreamOffset,
                CUDA_DEVICE_PTR ValueType *output, uint32_t outputOffset,
                const MyCdfLutPool &pool, // Must be a pool referencing GPU memory!
                MyRans rans)
                : bitstreamPtr(bitstream + bitstreamOffset), outputPtr(output + outputOffset),
                symbolLookup(pool), decoder(rans) {}

        CUDA_DEVICE void decode(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset, const uint32_t count) {
            const auto decoderId = getDecoderId<NInterleaved>();

            for (uint32_t i = 0; i + NInterleaved <= count; i += NInterleaved) {
                outputPtr[decoderId] = decodeOnce(cdfOffset, lutOffset);
                outputPtr += NInterleaved;
            }

            if (decoderId < count % NInterleaved) {
                outputPtr[decoderId] = decodeOnce(cdfOffset, lutOffset);
                outputPtr += count % NInterleaved;
            }
        }

        CUDA_DEVICE void decode(const CdfLutOffsetType allCdfOffsets[], const CdfLutOffsetType allLutOffsets[], const uint32_t count) {
            const auto decoderId = getDecoderId<NInterleaved>();

            uint32_t i = 0, cdfLutOffset = decoderId;
            for (; i + NInterleaved <= count; i += NInterleaved, outputPtr += NInterleaved, cdfLutOffset += NInterleaved) {
                outputPtr[decoderId] = decodeOnce(allCdfOffsets[cdfLutOffset], allLutOffsets[cdfLutOffset]);
            }

            if (decoderId < count % NInterleaved) {
                outputPtr[decoderId] = decodeOnce(allCdfOffsets[cdfLutOffset], allLutOffsets[cdfLutOffset]);
                outputPtr += count % NInterleaved;
            }
        }

    //protected:
        CUDA_DEVICE inline ValueType decodeOnce(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            auto value = decodeOnce_noRenorm(cdfOffset, lutOffset);
            renorm();

            return value;
        }

        CUDA_DEVICE inline ValueType decodeOnce_noRenorm(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            auto probability = decoder.decGetProbability();
            auto [value, start, frequency] = symbolLookup.getSymbolInfo(cdfOffset, lutOffset, probability);
            decoder.decAdvanceSymbol(start, frequency);

            return value;
        }

        /*
         * Return flag representing which threads renormalized.
         */
        CUDA_DEVICE inline uint32_t renorm() {
            bool shouldRenorm = decoder.decShouldRenorm();
            auto vote = __ballot_sync(0xffffffff, shouldRenorm);

            if (shouldRenorm) {
                auto offset = 1 - __popc(vote & getLaneMaskLe());
                decoder.decRenormOnce(bitstreamPtr[offset]);
            }

            auto amountRead = __popc(vote);
            bitstreamPtr -= amountRead;

            return vote;
        }

        MyRans decoder;
        MySymbolLookup symbolLookup;
        CUDA_DEVICE_PTR const RansBitstreamType * __restrict__ bitstreamPtr;
        CUDA_DEVICE_PTR ValueType * __restrict__ outputPtr;
    };
}

#endif //RECOIL_RANS_DECODER_CUDA_CUH
