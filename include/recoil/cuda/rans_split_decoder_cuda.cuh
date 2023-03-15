#ifndef RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
#define RECOIL_RANS_SPLIT_DECODER_CUDA_CUH

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/cuda/rans_decoder_cuda.cuh"
#include "recoil/cuda/macros.h"
#include "recoil/cuda/rans_coded_data_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda/std/array>
#include <chrono>
#include <iostream>

namespace Recoil {
    namespace {
        template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
                std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
                uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
                size_t NInterleaved, uint32_t NThreads>
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
                bool ransInitialized = false;
                uint32_t ransInitFlag = 0x00;

                for (uint32_t symbolGroupId = currentSplit.minSymbolGroupId; ransInitFlag != 0xffffffff; symbolGroupId++) {
                    if (!ransInitialized) {
                        if (currentSplit.startSymbolGroupIds[decoderId] == symbolGroupId) {
                            // Rans was not initialized, but current group is start symbol group
                            ransInitFlag |= decoder.renorm();
                            ransInitialized = true;
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
            }

            decoder.decode(cdfOffset, lutOffset, decodeEndSymbolId - decodeStartSymbolId);
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
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyRansSplitsMetadata = RansSplitsMetadata<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MySplitCuda = SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        RansSplitDecoderCuda(MyRansCodedData data, MyRansSplitsMetadata metadata, const MyCdfLutPool &pool)
            : data(std::move(data)), metadata(std::move(metadata)), pool(pool) {
            cudaCheck(cudaMalloc(&outputBuffer, sizeof(ValueType) * data.symbolCount));
            bitstream = allocAndCopyToGpu(this->data.bitstream.data(), sizeof(RansBitstreamType) * this->data.bitstream.size());

            std::vector<MySplitCuda> splitsLocal;
            for (const auto &split: this->metadata.splits) splitsLocal.emplace_back(split);
            splits = allocAndCopyToGpu(splitsLocal.data(), sizeof(MySplitCuda) * splitsLocal.size());

            poolBuf = allocAndCopyToGpu(pool.getPool(), pool.poolSize());
        }

        std::vector<ValueType> decodeAll(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            const int NThreads = 64;

            MyCdfLutPool poolGpu(
                    reinterpret_cast<const CdfType*>(reinterpret_cast<const uint8_t*>(pool.getCdfPool()) - pool.getPool() + poolBuf),
                    pool.getCdfSize(),
                    reinterpret_cast<const MyCdfLutPool::MyLutItem *>(reinterpret_cast<const uint8_t*>(pool.getLutPool()) - pool.getPool() + poolBuf),
                    pool.getLutSize()
            );

            cudaDeviceSynchronize();
            auto start = std::chrono::high_resolution_clock::now();

            launchCudaDecode_staticCdf<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved, NThreads>
                    <<<metadata.splits.size() / (NThreads / NInterleaved), NThreads>>>(
                    metadata.splits.size(), data.symbolCount, std::move(poolGpu),
                    bitstream, outputBuffer, splits, cdfOffset, lutOffset);
            cudaDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            lastDuration = duration.count();

            std::vector<ValueType> result;
            result.resize(data.symbolCount);
            cudaMemcpy(result.data(), outputBuffer, sizeof(ValueType) * data.symbolCount, cudaMemcpyDeviceToHost);

            return result;
        }

        ~RansSplitDecoderCuda() {
            cudaFree(bitstream);
            cudaFree(outputBuffer);
            cudaFree(splits);
            cudaFree(poolBuf);
        }

        uint64_t getLastDuration() { return lastDuration; }

    protected:
        MyRansCodedData data;
        MyRansSplitsMetadata metadata;
        const MyCdfLutPool &pool;

        CUDA_DEVICE_PTR ValueType *outputBuffer;
        CUDA_DEVICE_PTR RansBitstreamType *bitstream;
        CUDA_DEVICE_PTR MySplitCuda *splits;
        CUDA_DEVICE_PTR uint8_t *poolBuf;

        uint64_t lastDuration = 0;

        template<typename T>
        static T *allocAndCopyToGpu(const T *hostPtr, const size_t size) {
            CUDA_DEVICE_PTR T *devicePtr;
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&devicePtr), size));
            cudaCheck(cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice));

            return devicePtr;
        }

        static void cudaCheck(cudaError_t code) {
            if (code != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error: ") + std::string(cudaGetErrorString(code)));
            }
        }
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
