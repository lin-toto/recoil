#ifndef RECOIL_RANS_CUDA_LAUNCHER_CUH
#define RECOIL_RANS_CUDA_LAUNCHER_CUH

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/lib/cuda/macros.h"
#include "recoil/lib/cuda/cuda_libs.cuh"
#include "recoil/cuda/rans_decoder_cuda.cuh"
#include "recoil/cuda/rans_coded_data_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda/std/array>
#include <chrono>
#include <iostream>

namespace Recoil::CudaLauncher {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved>
    CUDA_DEVICE inline void syncRansOnce(
            RansDecoderCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity> &decoder,
            const SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved> &currentSplit,
            const uint32_t symbolGroupId, uint32_t &ransInitFlag,
            const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
        const int decoderId = threadIdx.x % NInterleaved;

        if (!(ransInitFlag & (1 << decoderId))) {
            if (currentSplit.startSymbolGroupIds[decoderId] == symbolGroupId) {
                // Rans was not initialized, but current group is start symbol group
                ransInitFlag |= decoder.renorm();
            } else {
                // Rans was not initialized, and still awaits initialization
                // Update bitstream pointer only depending on other threads.
                auto vote = __ballot_sync(0xffffffff, false);
                decoder.bitstreamPtr -= __popc(vote);
                ransInitFlag |= vote;
            }
        } else {
            // Perform normal decoding but do not record result.
            decoder.decodeOnce_noRenorm(cdfOffset, lutOffset);
            ransInitFlag |= decoder.renorm();
        }
    }

    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved, size_t NThreads>
    CUDA_GLOBAL void launchCudaDecode(
            const uint32_t nSplits,
            uint32_t totalSymbolCount,
            CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool,
            CUDA_DEVICE_PTR const RansBitstreamType *bitstream,
            CUDA_DEVICE_PTR ValueType *outputBuffer,
            CUDA_DEVICE_PTR SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved> *splits,
            const CdfLutOffsetType cdfOffset,
            const CdfLutOffsetType lutOffset
    ) {
        static_assert(NInterleaved == 32, "CUDA decoder only supports 32 interleaving");
        static_assert(NThreads % NInterleaved == 0, "NThreads must be multiples of NInterleaved");

        const auto splitId = getSplitId<NInterleaved, NThreads>(), decoderId = getDecoderId<NInterleaved>();
        const auto& currentSplit = splits[splitId];

        uint32_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId + 1);
        uint32_t decodeEndSymbolId = splitId == nSplits - 1 ? totalSymbolCount
                                                            : NInterleaved * (1 + splits[splitId + 1].maxSymbolGroupId);

        RansDecoderCuda decoder(
                bitstream, currentSplit.cutPosition,
                outputBuffer, decodeStartSymbolId, pool,
                currentSplit.intermediateRans[decoderId]);

        if (splitId != 0) {
            uint32_t ransInitFlag = 0x00;
            for (uint32_t symbolGroupId = currentSplit.minSymbolGroupId; ransInitFlag != 0xffffffff; symbolGroupId++) {
                syncRansOnce(decoder, currentSplit, symbolGroupId, ransInitFlag, cdfOffset, lutOffset);
            }
        }

        decoder.decode(cdfOffset, lutOffset, decodeEndSymbolId - decodeStartSymbolId);
    }

    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved, size_t NThreads>
    CUDA_GLOBAL void launchCudaDecode(
            const uint32_t nSplits,
            uint32_t totalSymbolCount,
            CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool,
            CUDA_DEVICE_PTR const RansBitstreamType *bitstream,
            CUDA_DEVICE_PTR ValueType *outputBuffer,
            CUDA_DEVICE_PTR SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved> *splits,
            const CdfLutOffsetType allCdfOffsets[],
            const CdfLutOffsetType allLutOffsets[]
    ) {
        static_assert(NInterleaved == 32, "CUDA decoder only supports 32 interleaving");
        static_assert(NThreads % NInterleaved == 0, "NThreads must be multiples of NInterleaved");

        const unsigned int splitId = blockIdx.x * (NThreads / NInterleaved) + threadIdx.x / NInterleaved, decoderId = threadIdx.x % NInterleaved;
        const auto& currentSplit = splits[splitId];

        uint32_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId + 1);
        uint32_t decodeEndSymbolId = splitId == nSplits - 1 ? totalSymbolCount
                                                            : NInterleaved * (1 + splits[splitId + 1].maxSymbolGroupId);

        RansDecoderCuda decoder(
                bitstream, currentSplit.cutPosition,
                outputBuffer, decodeStartSymbolId, pool,
                currentSplit.intermediateRans[decoderId]);

        if (splitId != 0) {
            uint32_t ransInitFlag = 0x00;
            for (uint32_t symbolGroupId = currentSplit.minSymbolGroupId; ransInitFlag != 0xffffffff; symbolGroupId++) {
                auto cdfLutOffset = NInterleaved * (symbolGroupId - currentSplit.minSymbolGroupId()) + decoderId;
                syncRansOnce(decoder, currentSplit, symbolGroupId, ransInitFlag,
                             allCdfOffsets[cdfLutOffset], allLutOffsets[cdfLutOffset]);
            }
        }

        auto cdfLutOffset = NInterleaved * (currentSplit.maxSymbolGroupId() + 1 - currentSplit.minSymbolGroupId());;
        decoder.decode(allCdfOffsets + cdfLutOffset, allLutOffsets + cdfLutOffset, decodeEndSymbolId - decodeStartSymbolId);
    }
}

#endif //RECOIL_RANS_CUDA_LAUNCHER_CUH
