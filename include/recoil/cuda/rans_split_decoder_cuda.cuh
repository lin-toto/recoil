#ifndef RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
#define RECOIL_RANS_SPLIT_DECODER_CUDA_CUH

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/cuda/rans_decoder_cuda.cuh"
#include "recoil/cuda/macros.h"

#include <cuda_runtime_api.h>
#include <cuda/std/array>
#include <chrono>

namespace Recoil {
    namespace {
        template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
                std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
                uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
                size_t NInterleaved>
        struct SplitCuda {
            using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;

            size_t cutPosition; // TODO: need to verify this is actually safe!
            cuda::std::array<MyRans, NInterleaved> intermediateRans;
            cuda::std::array<size_t, NInterleaved> startSymbolGroupIds;
            size_t minSymbolGroupId, maxSymbolGroupId;
        };

        template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
                std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
                uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
                size_t NInterleaved, size_t NSplits>
        CUDA_GLOBAL void launchCudaDecode_staticCdf(
                uint32_t totalSymbolCount,
                CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool,
                CUDA_DEVICE_PTR const RansBitstreamType *bitstream,
                CUDA_DEVICE_PTR ValueType *outputBuffer,
                CUDA_DEVICE_PTR SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved> *splits,
                const CdfLutOffsetType cdfOffset,
                const CdfLutOffsetType lutOffset
        ) {
            const unsigned int splitId = blockIdx.x, decoderId = threadIdx.x;
            const auto& currentSplit = splits[splitId];

            uint32_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId + 1);
            uint32_t decodeEndSymbolId = splitId == NSplits - 1 ? totalSymbolCount
                                                                : NInterleaved * (1 + splits[splitId + 1].maxSymbolGroupId);

            RansDecoderCuda decoder(
                    bitstream, currentSplit.cutPosition,
                    outputBuffer, decodeStartSymbolId, pool,
                    currentSplit.intermediateRans[decoderId]);

            if (splitId != 0) {
                bool ransInitialized = false;
                uint32_t ransInitFlag = 0x00;
                // FIXME: support other than 32 threads
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
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved, size_t NSplits>
    class RansSplitDecoderCuda {
        using MyRansCodedDataWithSplits = RansCodedDataWithSplits<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved, NSplits>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
    public:
        explicit RansSplitDecoderCuda(MyRansCodedDataWithSplits data, const MyCdfLutPool &pool) : data(std::move(data)), pool(pool) {}

        std::vector<ValueType> decodeAll(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            CUDA_DEVICE_PTR RansBitstreamType *bitstream;
            cudaMalloc(&bitstream, sizeof(RansBitstreamType) * data.bitstream.size());
            cudaMemcpy(bitstream, data.bitstream.data(), sizeof(RansBitstreamType) * data.bitstream.size(), cudaMemcpyHostToDevice);

            CUDA_DEVICE_PTR ValueType *outputBuffer;
            cudaMalloc(&outputBuffer, sizeof(ValueType) * data.symbolCount);

            std::array<SplitCuda<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved>, NSplits> splits;
            for (int i = 0; i < NSplits; i++) {
                splits[i].cutPosition = data.splits[i].cutPosition;
                std::copy(data.splits[i].intermediateRans.begin(), data.splits[i].intermediateRans.end(), splits[i].intermediateRans.begin());
                std::copy(data.splits[i].startSymbolGroupIds.begin(), data.splits[i].startSymbolGroupIds.end(), splits[i].startSymbolGroupIds.begin());
                splits[i].minSymbolGroupId = data.splits[i].minSymbolGroupId();
                splits[i].maxSymbolGroupId = data.splits[i].maxSymbolGroupId();
            }
            CUDA_DEVICE_PTR decltype(&splits[0]) splitsCuda;
            cudaMalloc(&splitsCuda, sizeof(splits[0]) * NSplits);
            cudaMemcpy(splitsCuda, &splits[0], sizeof(splits[0]) * NSplits, cudaMemcpyHostToDevice);

            CUDA_DEVICE_PTR uint8_t *poolBuf;
            cudaMalloc(&poolBuf, pool.poolSize());
            fflush(stdout);
            cudaMemcpy(poolBuf, pool.getPool(), pool.poolSize(), cudaMemcpyHostToDevice);
            MyCdfLutPool poolGpu(
                    reinterpret_cast<const CdfType*>(reinterpret_cast<const uint8_t*>(pool.getCdfPool()) - pool.getPool() + poolBuf),
                    pool.getCdfSize(),
                    reinterpret_cast<const MyCdfLutPool::MyLutItem *>(reinterpret_cast<const uint8_t*>(pool.getLutPool()) - pool.getPool() + poolBuf),
                    pool.getLutSize()
            );

            auto start = std::chrono::high_resolution_clock::now();

            launchCudaDecode_staticCdf<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved, NSplits>
                    <<<NSplits, NInterleaved>>>(
                    data.symbolCount, std::move(poolGpu), bitstream, outputBuffer, splitsCuda, cdfOffset, lutOffset);
            cudaDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            printf("time_us: %d\n", duration.count());

            std::vector<ValueType> result;
            result.resize(data.symbolCount);
            cudaMemcpy(result.data(), outputBuffer, sizeof(ValueType) * data.symbolCount, cudaMemcpyDeviceToHost);

            cudaFree(bitstream);
            cudaFree(outputBuffer);

            return result;
        }

    protected:
        MyRansCodedDataWithSplits data;
        const MyCdfLutPool &pool;
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
