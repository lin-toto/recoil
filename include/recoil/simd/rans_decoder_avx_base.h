#ifndef RECOIL_RANS_DECODER_AVX_BASE_H
#define RECOIL_RANS_DECODER_AVX_BASE_H

#include "recoil/rans_decoder.h"
#include "recoil/lib/simd/avx_datatypes.h"
#include <x86intrin.h>
#include <array>
#include <tuple>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved, SimdDataTypeWrapperConcept SimdDataTypeWrapper, typename SymbolLookupAVX>
    class RansDecoder_AVXBase : public RansDecoder<
            CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved> {
    protected:
        using MyRansDecoder = RansDecoder<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, LutGranularity, NInterleaved>;
        using SimdDataType = typename SimdDataTypeWrapper::SimdDataType;
        using MyRans = typename MyRansDecoder::MyRans;
        using MyCdfLutPool = typename MyRansDecoder::MyCdfLutPool;

        static constexpr size_t RansBatchSize = sizeof(SimdDataType) / sizeof(RansStateType);
        static constexpr size_t RansStepCount = NInterleaved / RansBatchSize;

        using SimdArrayType = std::array<RansStateType, RansBatchSize>;

        static_assert(NInterleaved % RansBatchSize == 0, "AVX decoder must work on RansPerBatchxN streams");
        static_assert(MyRansDecoder::MyRans::oneShotRenorm, "Only one shot renorm decoders are supported by AVX decoder");
    public:
        using MyRansDecoder::decode;

        RansDecoder_AVXBase(const std::span<RansBitstreamType> bitstream, std::array<MyRans, NInterleaved> rans, const MyCdfLutPool& pool)
            : symbolLookupAvx(pool), MyRansDecoder(bitstream, std::move(rans), pool) {}

        void decode(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset, const size_t count, const std::span<ValueType> output) override {
            if (output.size() < count) [[unlikely]] throw std::runtime_error("Not enough buffer space");
            size_t completedCount = 0;

            {
                // Step 1: decode initial unaligned parts so that ransIt is now at beginning
                auto unalignedCount = std::min(static_cast<size_t>(this->rans.end() - this->ransIt), count);
                if (unalignedCount != NInterleaved) { // If equal to NInterleaved, it is at beginning; no action needed
                    MyRansDecoder::decode(cdfOffset, lutOffset, unalignedCount, output);
                    completedCount += unalignedCount;
                }

                if (completedCount == count) [[unlikely]] return;
            }

            {
                // Step 2: decode aligned parts
                std::array<CdfLutOffsetType, NInterleaved> cdfOffsets, lutOffsets;
                cdfOffsets.fill(cdfOffset);
                lutOffsets.fill(lutOffset);

                SimdDataType ransSimds[RansStepCount];
                createRansSimds(ransSimds);
                for (; completedCount + NInterleaved <= count; completedCount += NInterleaved) {
                    decodeOnceAligned(cdfOffsets, lutOffsets,
                                      ransSimds, output.subspan(completedCount));
                }
                writeBackRansSimds(ransSimds);
            }

            {
                // Step 3: decode final unaligned parts
                if (completedCount != count) {
                    MyRansDecoder::decode(cdfOffset, lutOffset, count - completedCount, output.subspan(completedCount));
                }
            }
        }

        void decode(const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets, const std::span<ValueType> output) override {
            if (cdfOffsets.size() != lutOffsets.size()) [[unlikely]] throw std::runtime_error("CDF and LUT offset length mismatch");
            if (output.size() < cdfOffsets.size()) [[unlikely]] throw std::runtime_error("Not enough buffer space");
            if (!isAligned(cdfOffsets.data(), sizeof(SimdDataType)) || !isAligned(lutOffsets.data(), sizeof(SimdDataType)))
                throw std::runtime_error("Unaligned CDF or LUT offsets");
            size_t completedCount = 0, count = cdfOffsets.size();

            {
                // Step 1: decode initial unaligned parts so that ransIt is now at beginning
                auto unalignedCount = std::min(static_cast<size_t>(this->rans.end() - this->ransIt), count);
                if (unalignedCount != NInterleaved) { // If equal to NInterleaved, it is at beginning; no action needed
                    MyRansDecoder::decode(cdfOffsets.subspan(unalignedCount), lutOffsets.subspan(unalignedCount), output);
                    completedCount += unalignedCount;
                }

                if (completedCount == count) [[unlikely]] return;
            }

            {
                // Step 2: decode aligned parts
                SimdDataType ransSimds[RansStepCount];
                createRansSimds(ransSimds);
                for (; completedCount + NInterleaved <= count; completedCount += NInterleaved) {
                    decodeOnceAligned(cdfOffsets.subspan(completedCount),
                                      lutOffsets.subspan(completedCount),
                                      ransSimds, output.subspan(completedCount));
                }
                writeBackRansSimds(ransSimds);
            }

            {
                // Step 3: decode final unaligned parts
                if (completedCount != count) {
                    MyRansDecoder::decode(cdfOffsets.subspan(completedCount), lutOffsets.subspan(completedCount), output.subspan(completedCount));
                }
            }

        }
    protected:
        SymbolLookupAVX symbolLookupAvx;

        static inline bool isAligned(void *ptr, size_t align) {
            return !(reinterpret_cast<size_t>(ptr) % align);
        }

        inline void createRansSimds(SimdDataType ransSimds[]) {
            for (auto b = 0; b < RansStepCount; b++) {
                alignas(sizeof(SimdDataType)) SimdArrayType rans{};
                for (auto i = 0; i < RansBatchSize; i++) {
                    rans[i] = this->rans[b * RansBatchSize + i].state;
                }
                ransSimds[b] = SimdDataTypeWrapper::toSimd(rans);
            }
        }

        inline void writeBackRansSimds(SimdDataType ransSimds[]) {
            for (auto b = 0; b < RansStepCount; b++) {
                auto rans = SimdDataTypeWrapper::fromSimd(ransSimds[b]);
                for (auto i = 0; i < RansBatchSize; i++) {
                    this->rans[b * RansBatchSize + i].state = rans[i];
                }
            }
        }

        virtual void decodeOnceAligned(const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets,
                                       SimdDataType ransSimds[], const std::span<ValueType> output) {
            for (auto b = 0; b < RansStepCount; b++) {
                auto cdfOffsetsSimd = SimdDataTypeWrapper::toSimd(cdfOffsets.subspan(b * RansBatchSize).data());
                auto lutOffsetsSimd = SimdDataTypeWrapper::toSimd(lutOffsets.subspan(b * RansBatchSize).data());

                auto& ransSimd = ransSimds[b];
                auto probabilitiesSimd = getProbabilities(ransSimd);

                auto [symbolsSimd, startsSimd, frequenciesSimd] = symbolLookupAvx.getSymbolInfo(
                        cdfOffsetsSimd, lutOffsetsSimd, probabilitiesSimd);

                // TODO: if probability is a bypass sentinel, handle as bypass symbol

                advanceSymbol(ransSimd, probabilitiesSimd, startsSimd, frequenciesSimd);
                renormSimd(ransSimd);

                writeResult(symbolsSimd, output.data() + b * RansBatchSize);
            }
        }

        virtual void writeResult(const SimdDataType symbolsSimd, ValueType *ptr) {
            auto symbols = SimdDataTypeWrapper::fromSimd(symbolsSimd);
            std::copy(symbols.begin(), symbols.end(), ptr);
        }

        virtual void writeResult(const SimdDataType symbolsSimd1, const SimdDataType symbolsSimd2, ValueType *ptr) {
            writeResult(symbolsSimd1, ptr);
            writeResult(symbolsSimd2, ptr + sizeof(SimdDataType) / sizeof(RansStateType));
        }

        virtual SimdDataType getProbabilities(SimdDataType ransSimd) const = 0;
        virtual void advanceSymbol(SimdDataType &ransSimd, SimdDataType lastProbabilities,
                                   SimdDataType lastStarts, SimdDataType lastFrequencies) = 0;
        virtual void renormSimd(SimdDataType &ransSimd) = 0;
    };
}

#endif //RECOIL_RANS_DECODER_AVX_BASE_H
