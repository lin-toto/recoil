#ifndef RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
#define RECOIL_RANS_SPLIT_DECODER_CUDA_CUH

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include "recoil/cuda/rans_decoder_cuda.cuh"
#include "recoil/lib/cuda/macros.h"
#include "recoil/cuda/rans_coded_data_cuda.h"
#include "recoil/cuda/rans_cuda_launcher.cuh"

#include <cuda_runtime_api.h>
#include <cuda/std/array>
#include <chrono>

namespace Recoil {

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

        int estimateMaxOccupancySplits() {
            int maxBlocks;
            cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &maxBlocks,
                    CudaLauncher::launchCudaDecode_staticCdf<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved, NThreads>,
                    NThreads,
                    0));

            int deviceID;
            cudaDeviceProp props;

            cudaGetDevice(&deviceID);
            cudaGetDeviceProperties(&props, deviceID);
            return maxBlocks * props.multiProcessorCount * (NThreads / NInterleaved);
        }

        std::vector<ValueType> decodeAll(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            auto start = std::chrono::high_resolution_clock::now();

            CudaLauncher::launchCudaDecode_staticCdf<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved, NThreads>
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

        std::vector<ValueType> decodeAll(const std::span<CdfLutOffsetType> allCdfOffsets, const std::span<CdfLutOffsetType> allLutOffsets) {
            if (allCdfOffsets.size() != allLutOffsets.size()) [[unlikely]] throw std::runtime_error("CDF and LUT offset length mismatch");
            if (allCdfOffsets.size() != data.symbolCount) [[unlikely]] throw std::runtime_error("Need the full CDF");

            CUDA_DEVICE_PTR auto *allCdfOffsetsCuda = allocAndCopyToGpu(allCdfOffsets), *allLutOffsetsCuda = allocAndCopyToGpu(allLutOffsets);

            auto start = std::chrono::high_resolution_clock::now();

            CudaLauncher::launchCudaDecode_multiCdf<<<metadata.splits.size() / (NThreads / NInterleaved), NThreads>>>(
                    metadata.splits.size(), data.symbolCount, std::move(poolGpu), bitstream, outputBuffer, splits, allCdfOffsetsCuda, allLutOffsetsCuda);
            cudaDeviceSynchronize();

            cudaFree(allCdfOffsetsCuda);
            cudaFree(allLutOffsetsCuda);

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

        template<typename T>
        static T *allocAndCopyToGpu(const T *hostPtr, const size_t size) {
            CUDA_DEVICE_PTR T *devicePtr;
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&devicePtr), size));
            cudaCheck(cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice));

            return devicePtr;
        }

        template<typename T>
        static T *allocAndCopyToGpu(const std::span<T> span) {
            return allocAndCopyToGpu(span.data(), span.size_bytes());
        }

        static void cudaCheck(cudaError_t code) {
            if (code != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error: ") + std::string(cudaGetErrorString(code)));
            }
        }
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
