#ifndef RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
#define RECOIL_RANS_SPLIT_DECODER_CUDA_CUH

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/cuda/rans_decoder_cuda.cuh"
#include "recoil/lib/cuda/macros.h"
#include "recoil/cuda/rans_coded_data_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda/std/array>
#include <chrono>

namespace Recoil {
    namespace SplitDecoderCuda {
        template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
                std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
                uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
                size_t NInterleaved>
        CUDA_DEVICE inline void syncRansOnce(
                RansDecoderCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity> &decoder,
                const SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved> &currentSplit,
                const uint32_t symbolGroupId, uint32_t &ransInitFlag,
                const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            const int decoderId = getDecoderId<NInterleaved>();

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
        CUDA_GLOBAL void launchCudaDecode_staticCdf(
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
            if (splitId >= nSplits) return;
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
        CUDA_GLOBAL void launchCudaDecode_multiCdf(
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

            const unsigned int splitId = getSplitId<NInterleaved, NThreads>(), decoderId = getDecoderId<NInterleaved>();
            if (splitId >= nSplits) return;
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

            auto cdfLutOffset = NInterleaved * (currentSplit.maxSymbolGroupId() + 1 - currentSplit.minSymbolGroupId());

            decoder.decode(allCdfOffsets + cdfLutOffset, allLutOffsets + cdfLutOffset, decodeEndSymbolId - decodeStartSymbolId);
        }
    }

    /*
     * CUDA interface class between host and device code.
     * From this point the host containers will be converted to CUDA equivalents.
     */
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity, size_t NInterleaved>
    class RansSplitDecoderCuda {
        static const int NThreads = 128;

        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyRansSplitsMetadata = RansSplitsMetadata<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MySplitCuda = SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        RansSplitDecoderCuda(MyRansCodedData data, MyRansSplitsMetadata metadata, const MyCdfLutPool &pool)
            : data(std::move(data)), metadata(std::move(metadata)),
            bitstream(allocAndCopyToGpu(this->data.getRealBitstream())),
            poolBuf(allocAndCopyToGpu(pool.getPool(), pool.poolSize())),
            poolGpu(reinterpret_cast<const CdfType*>(reinterpret_cast<const uint8_t*>(pool.getCdfPool()) - pool.getPool() + poolBuf),
                    pool.getCdfSize(),
                    reinterpret_cast<const MyCdfLutPool::MyLutItem *>(reinterpret_cast<const uint8_t*>(pool.getLutPool()) - pool.getPool() + poolBuf),
                    pool.getLutSize()) {
            cudaCheck(cudaMalloc(&outputBuffer, sizeof(ValueType) * data.symbolCount));

            std::vector<MySplitCuda> splitsLocal;
            for (const auto &split: this->metadata.splits) splitsLocal.emplace_back(split);
            splits = allocAndCopyToGpu(std::span{splitsLocal});
        }

        static int estimateMaxOccupancySplits() {
            return estimateMaxOccupancy(
                    reinterpret_cast<const void*>(SplitDecoderCuda::launchCudaDecode_staticCdf<
                            CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound,
                            WriteBits, LutGranularity, NInterleaved, NThreads>),
                    NThreads) * (NThreads / NInterleaved);
        }

        std::vector<ValueType> decodeAll(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            SplitDecoderCuda::launchCudaDecode_staticCdf<
                    CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound,
                    WriteBits, LutGranularity, NInterleaved, NThreads>
            <<<metadata.splits.size() / (NThreads / NInterleaved), NThreads>>>(
                    metadata.splits.size(), data.symbolCount, std::move(poolGpu),
                    bitstream, outputBuffer, splits, cdfOffset, lutOffset);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            lastDuration = milliseconds * 1000;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            std::vector<ValueType> result;
            result.resize(data.symbolCount);
            cudaMemcpy(result.data(), outputBuffer, sizeof(ValueType) * data.symbolCount, cudaMemcpyDeviceToHost);

            return result;
        }

        std::vector<ValueType> decodeAll(const std::span<CdfLutOffsetType> allCdfOffsets, const std::span<CdfLutOffsetType> allLutOffsets) {
            if (allCdfOffsets.size() != allLutOffsets.size()) [[unlikely]] throw std::runtime_error("CDF and LUT offset length mismatch");
            if (allCdfOffsets.size() != data.symbolCount) [[unlikely]] throw std::runtime_error("Need the full CDF");

            CUDA_DEVICE_PTR auto *allCdfOffsetsCuda = allocAndCopyToGpu(allCdfOffsets), *allLutOffsetsCuda = allocAndCopyToGpu(allLutOffsets);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            SplitDecoderCuda::launchCudaDecode_multiCdf<
                    CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound,
                    WriteBits, LutGranularity, NInterleaved, NThreads>
            <<<metadata.splits.size() / (NThreads / NInterleaved), NThreads>>>(
                    metadata.splits.size(), data.symbolCount, std::move(poolGpu),
                    bitstream, outputBuffer, splits, allCdfOffsetsCuda, allLutOffsetsCuda);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            lastDuration = milliseconds * 1000;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaFree(allCdfOffsetsCuda);
            cudaFree(allLutOffsetsCuda);

            std::vector<ValueType> result;
            result.resize(data.symbolCount);
            cudaMemcpy(result.data(), outputBuffer, sizeof(ValueType) * data.symbolCount, cudaMemcpyDeviceToHost);

            return result;
        }

        ~RansSplitDecoderCuda() {
            cudaFree(bitstream);
            cudaFree(splits);
            cudaFree(poolBuf);
            cudaFree(outputBuffer);
        }

        uint64_t getLastDuration() { return lastDuration; }

    protected:
        MyRansCodedData data;
        MyRansSplitsMetadata metadata;

        CUDA_DEVICE_PTR ValueType *outputBuffer;
        CUDA_DEVICE_PTR RansBitstreamType *bitstream;
        CUDA_DEVICE_PTR MySplitCuda *splits;
        CUDA_DEVICE_PTR uint8_t *poolBuf;

        MyCdfLutPool poolGpu;

        uint64_t lastDuration = 0;
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
