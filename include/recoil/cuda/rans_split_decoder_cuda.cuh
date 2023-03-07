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
        template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
                BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
                size_t NInterleaved>
        struct SplitCuda {
            using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;

            size_t cutPosition; // TODO: need to verify this is actually safe!
            cuda::std::array<MyRans, NInterleaved> intermediateRans;
            cuda::std::array<size_t, NInterleaved> startSymbolGroupIds;
            size_t minSymbolGroupId, maxSymbolGroupId;
        };

        template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
                BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
                size_t NInterleaved, size_t NSplits>
        CUDA_GLOBAL void launchCudaDecode_staticCdf(
                uint32_t totalSymbolCount,
                CUDA_DEVICE_PTR RansBitstreamType *bitstream,
                CUDA_DEVICE_PTR ValueType *outputBuffer,
                CUDA_DEVICE_PTR SplitCuda<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved> *splits,
                CUDA_DEVICE_PTR CdfType *cdf,
                CUDA_DEVICE_PTR ValueType *lut
        ) {
            const unsigned int splitId = blockIdx.x, decoderId = threadIdx.x;
            const auto& currentSplit = splits[splitId];

            uint32_t decodeStartSymbolId = splitId == 0 ? 0 : NInterleaved * (currentSplit.maxSymbolGroupId + 1);
            uint32_t decodeEndSymbolId = splitId == NSplits - 1 ? totalSymbolCount
                                                                : NInterleaved * (1 + splits[splitId + 1].maxSymbolGroupId);

            RansDecoderCuda decoder(
                    bitstream, currentSplit.cutPosition,
                    outputBuffer, decodeStartSymbolId,
                    currentSplit.intermediateRans[decoderId]);

            if (splitId != 0) {
                bool ransInitialized = false, ransAllInitialized = false;
                for (uint32_t symbolGroupId = currentSplit.minSymbolGroupId; !ransAllInitialized; symbolGroupId++) {
                    bool shouldRenorm = false;
                    if (!ransInitialized) {
                        if (currentSplit.startSymbolGroupIds[decoderId] == symbolGroupId) {
                            ransInitialized = true;
                            shouldRenorm = true;
                        }
                    } else {
                        auto probability = decoder.decoder.decGetProbability();
                        auto symbol = lut[probability];
                        //printf("threadid %d sym %c\n", threadIdx.x, symbol);

                        // TODO: add LUT + CDF mixed support
                        auto start = cdf[symbol];
                        auto frequency = cdf[symbol + 1] - cdf[symbol];
                        decoder.decoder.decAdvanceSymbol(start, frequency);
                        shouldRenorm = true;
                    }

                    bool shouldRenorm_ = decoder.decoder.decShouldRenorm() && shouldRenorm;
                    // TODO: fix ballot_sync flag to support other than 32 threads
                    auto vote = __ballot_sync(0xffffffff, shouldRenorm_);

                    if (shouldRenorm_) {
                        auto offset = 1 - __popc(vote & getLaneMaskLe());
                        decoder.decoder.decRenormOnce(decoder.bitstreamPtr[offset]);
                        //printf("threadid %d renorm read from %d\n", threadIdx.x, offset);
                    }

                    auto amountRead = __popc(vote);
                    decoder.bitstreamPtr -= amountRead;

                    // FIXME: support other than 32 threads
                    ransAllInitialized = __ballot_sync(0xffffffff, ransInitialized) == 0xffffffff;
                }
            }

            decoder.decode(cdf, lut, decodeEndSymbolId - decodeStartSymbolId);
        }
    }

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
        explicit RansSplitDecoderCuda(MyRansCodedDataWithSplits data) : data(std::move(data)) {}

        // TODO: proper CDF store
        std::vector<ValueType> decodeAll(CUDA_DEVICE_PTR CdfType *cdf, CUDA_DEVICE_PTR ValueType *lut) {
            CUDA_DEVICE_PTR RansBitstreamType *bitstream;
            cudaMalloc(&bitstream, sizeof(RansBitstreamType) * data.bitstream.size());
            cudaMemcpy(bitstream, data.bitstream.data(), sizeof(RansBitstreamType) * data.bitstream.size(), cudaMemcpyHostToDevice);

            CUDA_DEVICE_PTR ValueType *outputBuffer;
            cudaMalloc(&outputBuffer, sizeof(ValueType) * data.symbolCount);

            std::array<SplitCuda<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>, NSplits> splits;
            for (int i = 0; i < NSplits; i++) {
                splits[i].cutPosition = data.splits[i].cutPosition;
                std::copy(data.splits[i].intermediateRans.begin(), data.splits[i].intermediateRans.end(), splits[i].intermediateRans.begin());
                std::copy(data.splits[i].startSymbolGroupIds.begin(), data.splits[i].startSymbolGroupIds.end(), splits[i].startSymbolGroupIds.begin());
                splits[i].minSymbolGroupId = data.splits[i].minSymbolGroupId();
                splits[i].maxSymbolGroupId = data.splits[i].maxSymbolGroupId();
            }
            CUDA_DEVICE_PTR SplitCuda<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved> *splitsCuda;
            cudaMalloc(&splitsCuda, sizeof(splits[0]) * NSplits);
            cudaMemcpy(splitsCuda, &splits[0], sizeof(splits[0]) * NSplits, cudaMemcpyHostToDevice);

            auto start = std::chrono::high_resolution_clock::now();

            launchCudaDecode_staticCdf<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved, NSplits>
                    <<<NSplits, NInterleaved>>>(
                    data.symbolCount, bitstream, outputBuffer, splitsCuda, cdf, lut);
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
    };
}

#endif //RECOIL_RANS_SPLIT_DECODER_CUDA_CUH
