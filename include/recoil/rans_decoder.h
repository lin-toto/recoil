#ifndef RECOIL_RANS_DECODER_H
#define RECOIL_RANS_DECODER_H

#include "recoil/lib/cdf.h"
#include "recoil/rans.h"
#include <vector>
#include <span>
#include <array>

namespace Recoil {
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            unsigned int NInterleaved>
    class RansDecoder {
    protected:
        using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
    public:
        RansDecoder(const std::span<RansBitstreamType> bitstream, std::array<MyRans, NInterleaved> rans)
                : rans(std::move(rans)), bitstream(bitstream),
                  ransIt(this->rans.begin()), bitstreamReverseIt(this->bitstream.rbegin()) {}

        /*
         * Decode all available values with a single shared CDF.
         */
        std::vector<ValueType> decode(const Cdf cdf) {
            std::vector<ValueType> result;
            // TODO

            return result;
        }

        /*
         * Decode count number of values with a single shared CDF.
         */
        std::vector<ValueType> decode(const Cdf cdf, const size_t count) {
            std::vector<ValueType> result;
            result.reserve(count);

            for (auto i = 0; i < count; i++) {
                result.push_back(decodeSymbol(*ransIt, cdf));

                if constexpr (NInterleaved > 1) {
                    ransIt++;
                    if (ransIt == rans.end()) ransIt = rans.begin();
                }
            }

            return result;
        }

        /*
         * Decode the values with independent CDF for each symbol.
         */
        std::vector<ValueType> decode(const std::span<Cdf> cdfs) {
            std::vector<ValueType> result;
            result.reserve(cdfs.size());

            for (auto& cdf : cdfs) {
                result.push_back(decodeSymbol(ransIt, cdf));

                if constexpr (NInterleaved > 1) {
                    ransIt++;
                    if (ransIt == rans.end()) ransIt = rans.begin();
                }
            }

            return result;
        }

    protected:
        std::array<MyRans, NInterleaved> rans;
        std::span<RansBitstreamType> bitstream;

        typename std::array<MyRans, NInterleaved>::iterator ransIt;
        typename std::span<RansBitstreamType>::reverse_iterator bitstreamReverseIt;

        ValueType decodeSymbol(MyRans &decoder, const Cdf cdf) {
            auto probability = decoder.decGetProbability();

            auto symbol = cdf.findValue(probability);
            if (symbol.has_value()) [[likely]] {
                auto [start, frequency] = cdf.getStartAndFrequency(symbol.value()).value();
                renorm(decoder, start, frequency);
                return symbol.value();
            } else {
                // TODO: if probability is a bypass sentinel, handle as bypass symbol
            }
        }

        void renorm(MyRans &decoder, const CdfType lastStart, const CdfType lastFrequency) {
            decoder.decAdvanceSymbol(lastStart, lastFrequency);

            if (bitstreamReverseIt == bitstream.rend()) [[unlikely]] {
                // TODO: notify outside about bitstream end, so decode can be terminated
                return;
            }

            if constexpr (MyRans::oneShotRenorm) {
                bool renormed = decoder.decRenormOnce(*bitstreamReverseIt);
                if (renormed) bitstreamReverseIt++;
            } else {
                bool renormed = decoder.decRenormOnce(*bitstreamReverseIt);
                while (renormed) {
                    bitstreamReverseIt++;
                    renormed = decoder.decRenormOnce(*bitstreamReverseIt);
                }
            }
        }
    };
}

#endif //RECOIL_RANS_DECODER_H
