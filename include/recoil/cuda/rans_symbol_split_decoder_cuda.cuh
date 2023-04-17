#ifndef RECOIL_RANS_SYMBOL_SPLIT_DECODER_CUDA_CUH
#define RECOIL_RANS_SYMBOL_SPLIT_DECODER_CUDA_CUH

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/cuda/rans_decoder_cuda.cuh"
#include "recoil/lib/cuda/macros.h"
#include "recoil/lib/math.h"
#include "recoil/cuda/rans_coded_data_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda/std/array>
#include <chrono>
#include <numeric>

namespace Recoil {
    namespace SymbolSplitDecoderCuda {
        template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
                std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
                uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
                size_t NInterleaved, size_t NThreads>
        CUDA_GLOBAL void launchCudaDecode_staticCdf(
                const uint32_t nSplits,
                uint32_t totalSymbolCount,
                CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool,
                CUDA_DEVICE_PTR const RansBitstreamType *bitstreams,
                CUDA_DEVICE_PTR const uint32_t *bitstreamOffsets,
                CUDA_DEVICE_PTR const Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits> *rans,
                CUDA_DEVICE_PTR ValueType *outputBuffer,
                const CdfLutOffsetType cdfOffset,
                const CdfLutOffsetType lutOffset
        ) {
            static_assert(NInterleaved == 32, "CUDA decoder only supports 32 interleaving");
            static_assert(NThreads % NInterleaved == 0, "NThreads must be multiples of NInterleaved");

            const auto perSplitSymbolCount = saveDiv<uint32_t>(totalSymbolCount, nSplits);
            const auto splitId = getSplitId<NInterleaved, NThreads>(), decoderId = getDecoderId<NInterleaved>();
            if (splitId >= nSplits) return;

            RansDecoderCuda decoder(
                    bitstreams, bitstreamOffsets[splitId],
                    outputBuffer, perSplitSymbolCount * splitId, pool,
                    rans[splitId * NInterleaved + decoderId]);

            decoder.decode(cdfOffset, lutOffset, splitId == nSplits - 1 ? totalSymbolCount - (perSplitSymbolCount * (nSplits - 1)) : perSplitSymbolCount);
        }

        template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
                std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
                uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
                size_t NInterleaved, size_t NThreads>
        CUDA_GLOBAL void launchCudaDecode_multiCdf(
                const uint32_t nSplits,
                uint32_t totalSymbolCount,
                CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool,
                CUDA_DEVICE_PTR const RansBitstreamType *bitstreams,
                CUDA_DEVICE_PTR const uint32_t *bitstreamOffsets,
                CUDA_DEVICE_PTR const Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits> *rans,
                CUDA_DEVICE_PTR ValueType *outputBuffer,
                const CdfLutOffsetType allCdfOffsets[],
                const CdfLutOffsetType allLutOffsets[]
        ) {
            static_assert(NInterleaved == 32, "CUDA decoder only supports 32 interleaving");
            static_assert(NThreads % NInterleaved == 0, "NThreads must be multiples of NInterleaved");

            const auto perSplitSymbolCount = saveDiv<uint32_t>(totalSymbolCount, nSplits);
            const auto splitId = getSplitId<NInterleaved, NThreads>(), decoderId = getDecoderId<NInterleaved>();
            if (perSplitSymbolCount * splitId >= totalSymbolCount) return;

            RansDecoderCuda decoder(
                    bitstreams, bitstreamOffsets[splitId],
                    outputBuffer, perSplitSymbolCount * splitId, pool,
                    rans[splitId * NInterleaved + decoderId]);

            decoder.decode(allCdfOffsets + perSplitSymbolCount * splitId,
                           allLutOffsets + perSplitSymbolCount * splitId,
                           perSplitSymbolCount * (splitId + 1) >= totalSymbolCount ?
                                totalSymbolCount - perSplitSymbolCount * splitId : perSplitSymbolCount);
        }
    }

    /*
     * CUDA interface class between host and device code.
     * From this point the host containers will be converted to CUDA equivalents.
     */
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity, size_t NInterleaved>
    class RansSymbolSplitDecoderCuda {
        static const int NThreads = 128;

        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
    public:
        RansSymbolSplitDecoderCuda(std::vector<MyRansCodedData> data, const MyCdfLutPool &pool)
            : data(std::move(data)),
            totalSymbolCount(std::accumulate(this->data.begin(), this->data.end(), 0, [](size_t len, auto &d) { return len + d.symbolCount; })),
            poolBuf(allocAndCopyToGpu(pool.getPool(), pool.poolSize())),
            poolGpu(reinterpret_cast<const CdfType*>(reinterpret_cast<const uint8_t*>(pool.getCdfPool()) - pool.getPool() + poolBuf),
                    pool.getCdfSize(),
                    reinterpret_cast<const MyCdfLutPool::MyLutItem *>(reinterpret_cast<const uint8_t*>(pool.getLutPool()) - pool.getPool() + poolBuf),
                    pool.getLutSize()) {
            cudaCheck(cudaMalloc(&outputBuffer, sizeof(ValueType) * totalSymbolCount));
            cudaCheck(cudaMalloc(&bitstreamOffsets, sizeof(uint32_t) * this->data.size()));
            cudaCheck(cudaMalloc(&finalRans, sizeof(RansStateType) * NInterleaved * this->data.size()));

            size_t totalBitstreamLength = std::accumulate(this->data.begin(), this->data.end(), 0, [](size_t len, auto &d) { return len + d.getRealBitstream().size(); });
            cudaCheck(cudaMalloc(&bitstreams, sizeof(RansBitstreamType) * totalBitstreamLength));
            auto *bitstreamsPtr = bitstreams;
            for (int splitId = 0; splitId < this->data.size(); splitId++) {
                auto &d = this->data[splitId];
                cudaMemcpy(bitstreamsPtr, d.getRealBitstream().data(), d.getRealBitstream().size_bytes(), cudaMemcpyHostToDevice);

                bitstreamsPtr += d.getRealBitstream().size();

                unsigned int offset = bitstreamsPtr - bitstreams - 1;
                cudaMemcpy(&bitstreamOffsets[splitId], &offset, sizeof(unsigned int), cudaMemcpyHostToDevice);

                cudaMemcpy(finalRans + splitId * NInterleaved, d.finalRans.data(), d.finalRans.size() * sizeof(RansStateType), cudaMemcpyHostToDevice);
            }
        }

        int estimateMaxOccupancySplits() {
            return estimateMaxOccupancy(
                    reinterpret_cast<const void*>(SymbolSplitDecoderCuda::launchCudaDecode_staticCdf<
                            CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound,
                            WriteBits, LutGranularity, NInterleaved, NThreads>),
                    NThreads) * (NThreads / NInterleaved);
        }

        std::vector<ValueType> decodeAll(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            SymbolSplitDecoderCuda::launchCudaDecode_staticCdf<
                    CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound,
                    WriteBits, LutGranularity, NInterleaved, NThreads>
            <<<data.size() / (NThreads / NInterleaved), NThreads>>>(
                    data.size(), totalSymbolCount, std::move(poolGpu),
                    bitstreams, bitstreamOffsets, finalRans, outputBuffer, cdfOffset, lutOffset);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            lastDuration = milliseconds * 1000;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            std::vector<ValueType> result;
            result.resize(totalSymbolCount);
            cudaMemcpy(result.data(), outputBuffer, sizeof(ValueType) * totalSymbolCount, cudaMemcpyDeviceToHost);

            return result;
        }

        std::vector<ValueType> decodeAll(const std::span<CdfLutOffsetType> allCdfOffsets, const std::span<CdfLutOffsetType> allLutOffsets) {
            if (allCdfOffsets.size() != allLutOffsets.size()) [[unlikely]] throw std::runtime_error("CDF and LUT offset length mismatch");

            CUDA_DEVICE_PTR const auto *allCdfOffsetsCuda = allocAndCopyToGpu(allCdfOffsets), *allLutOffsetsCuda = allocAndCopyToGpu(allLutOffsets);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            SymbolSplitDecoderCuda::launchCudaDecode_multiCdf<
                    CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound,
                    WriteBits, LutGranularity, NInterleaved, NThreads>
            <<<data.size() / (NThreads / NInterleaved), NThreads>>>(
                    data.size(), totalSymbolCount, std::move(poolGpu),
                    bitstreams, bitstreamOffsets, finalRans, outputBuffer, allCdfOffsetsCuda, allLutOffsetsCuda);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            lastDuration = milliseconds * 1000;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            std::vector<ValueType> result;
            result.resize(totalSymbolCount);
            cudaMemcpy(result.data(), outputBuffer, sizeof(ValueType) * totalSymbolCount, cudaMemcpyDeviceToHost);

            return result;
        }

        ~RansSymbolSplitDecoderCuda() {
            cudaFree(bitstreams);
            cudaFree(bitstreamOffsets);
            cudaFree(finalRans);
            cudaFree(poolBuf);
            cudaFree(outputBuffer);
        }

        uint64_t getLastDuration() { return lastDuration; }

    protected:
        std::vector<MyRansCodedData> data;
        size_t totalSymbolCount;

        CUDA_DEVICE_PTR ValueType *outputBuffer;
        CUDA_DEVICE_PTR uint8_t *poolBuf;
        CUDA_DEVICE_PTR RansBitstreamType *bitstreams;
        CUDA_DEVICE_PTR uint32_t *bitstreamOffsets;
        CUDA_DEVICE_PTR MyRans *finalRans;

        MyCdfLutPool poolGpu;

        uint64_t lastDuration = 0;
    };
}

#endif //RECOIL_RANS_SYMBOL_SPLIT_DECODER_CUDA_CUH
