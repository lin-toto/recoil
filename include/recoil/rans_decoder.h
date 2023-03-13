#ifndef RECOIL_RANS_DECODER_H
#define RECOIL_RANS_DECODER_H

#include "recoil/rans.h"
#include <vector>
#include <span>
#include <array>
#include <exception>

namespace Recoil {
    class DecodingReachesEndException : public std::exception {};

    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved>
    class RansDecoder {
        template<std::unsigned_integral, std::unsigned_integral, std::unsigned_integral T, std::unsigned_integral, uint8_t, T, uint8_t, uint8_t, size_t, size_t>
        friend class RansSplitDecoder;
    protected:
        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MySymbolLookup = SymbolLookup<CdfType, ValueType, ProbBits, LutGranularity>;
    public:
        RansDecoder(const std::span<RansBitstreamType> bitstream, std::array<MyRans, NInterleaved> rans, const MyCdfLutPool& pool)
                : rans(std::move(rans)), bitstream(bitstream), symbolLookup(pool),
                  ransIt(this->rans.begin()), bitstreamReverseIt(this->bitstream.rbegin()) {}

        /*
         * Decode count number of values with a single shared CDF.
         */
        virtual void decode(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset, const size_t count, const std::span<ValueType> output) {
            if (output.size() < count) [[unlikely]] throw std::runtime_error("Not enough buffer space");

            for (auto i = 0; i < count; i++) {
                output[i] = decodeSymbol(*ransIt, cdfOffset, lutOffset);

                if constexpr (NInterleaved > 1) {
                    ransIt++;
                    if (ransIt == rans.end()) ransIt = rans.begin();
                }
            }
        }

        std::vector<ValueType> decode(const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset, const size_t count) {
            std::vector<ValueType> result;
            result.resize(count);
            decode(cdfOffset, lutOffset, count, std::span{result});
            return result;
        }

        /*
         * Decode the values with independent CDF for each symbol.
         */
        virtual void decode(const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets, const std::span<ValueType> output) {
            if (cdfOffsets.size() != lutOffsets.size()) [[unlikely]] throw std::runtime_error("CDF and LUT offset length mismatch");
            if (output.size() < cdfOffsets.size()) [[unlikely]] throw std::runtime_error("Not enough buffer space");

            for (int i = 0; i < cdfOffsets.size(); i++) {
                output[i] = decodeSymbol(*ransIt, cdfOffsets[i], lutOffsets[i]);

                if constexpr (NInterleaved > 1) {
                    ransIt++;
                    if (ransIt == rans.end()) ransIt = rans.begin();
                }
            }
        }

        std::vector<ValueType> decode(const std::span<CdfLutOffsetType> cdfOffsets, const std::span<CdfLutOffsetType> lutOffsets) {
            std::vector<ValueType> result;
            result.resize(cdfOffsets.size());
            decode(cdfOffsets, lutOffsets, std::span{result});
            return result;
        }

    protected:
        std::array<MyRans, NInterleaved> rans;
        MySymbolLookup symbolLookup;
        std::span<RansBitstreamType> bitstream;

        typename std::array<MyRans, NInterleaved>::iterator ransIt;
        typename std::span<RansBitstreamType>::reverse_iterator bitstreamReverseIt;

        inline ValueType decodeSymbol(MyRans &decoder, const CdfLutOffsetType cdfOffset, const CdfLutOffsetType lutOffset) {
            auto probability = decoder.decGetProbability();

            auto [value, start, frequency] = symbolLookup.getSymbolInfo(cdfOffset, lutOffset, probability);
            decoder.decAdvanceSymbol(start, frequency);
            renorm(decoder);
            return value;

            // TODO: if probability is a bypass sentinel, handle as bypass symbol
        }

        inline void renorm(MyRans &decoder) {
            if (bitstreamReverseIt == bitstream.rend()) {
                if (decoder.state < RenormLowerBound) [[unlikely]] {
                    throw DecodingReachesEndException();
                } else return;
            } else {
                if constexpr (MyRans::oneShotRenorm) {
                    if (decoder.decShouldRenorm()) {
                        decoder.decRenormOnce(*bitstreamReverseIt);
                        bitstreamReverseIt++;
                    }
                } else {
                    while (decoder.decShouldRenorm()) {
                        decoder.decRenormOnce(*bitstreamReverseIt);
                        bitstreamReverseIt++;
                    }
                }
            }
        }
    };
}

#endif //RECOIL_RANS_DECODER_H
