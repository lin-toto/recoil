#ifndef RECOIL_RANS_ENCODER_H
#define RECOIL_RANS_ENCODER_H

#include "recoil/lib/cdf.h"
#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include <vector>
#include <span>
#include <bit>
#include <string>
#include <variant>
#include <cassert>

namespace Recoil {
    template<typename RansStateType, typename RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t NInterleaved>
    class RansEncoder {
        using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MyRansCodedData = RansCodedData<
                RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        explicit RansEncoder(std::array<MyRans, NInterleaved> rans) : rans(std::move(rans)) {}

        /*
         * Buffer the values for encode with single shared CDF.
         */
        void buffer(const std::span<ValueType> values, const Cdf cdf) {
            symbolBuffer.reserve(symbolBuffer.size() + values.size());

            for (const auto &value: values) {
                bufferSymbol(value, cdf);
            }
        }

        /*
         * Buffer the values for encode with independent CDF for each symbol.
         */
        void buffer(const std::span<ValueType> values, const std::span<Cdf> cdfs) {
            assert(values.size() == cdfs.size());
            symbolBuffer.reserve(symbolBuffer.size() + values.size());

            for (auto i = 0; i < values.size(); i++) {
                bufferSymbol(values[i], cdfs[i]);
            }
        }

        /*
         * Flush the buffered values into the bitstream.
         *
         * We encode the symbols in reverse order so decoder produces them in the normal order.
         * In interleaved rANS, we start with the last decoder if (nSymbols - 1) % NInterleaved == 0,
         * but when not so, align it with the correct entropy coder.
         * We should be sending it to coder (NInterleaved - 1) - (NInterleaved - (nSymbols - 1) % NInterleaved - 1).
         *
         * For example:
         * Coders 0 1 2 3
         * Input  0 1 2 3
         *        4 5 6
         * During encode we start with 6, so it goes to coder 3 - (4 - 6 % 4 - 1) = 2.
         */
        MyRansCodedData flush() {
            auto ransIt = rans.rbegin();
            if constexpr (NInterleaved > 1) ransIt += NInterleaved - (symbolBuffer.size() - 1) % NInterleaved - 1;
            for (auto symbol = symbolBuffer.rbegin(); symbol != symbolBuffer.rend(); symbol++) {
                encodeSymbol(*ransIt, *symbol);

                if constexpr (NInterleaved > 1) {
                    ransIt++;
                    if (ransIt == rans.rend()) ransIt = rans.rbegin();
                }
            }

            MyRansCodedData result{bitstream, rans};

            bitstream.clear();
            symbolBuffer.clear();
            std::for_each(rans.begin(), rans.end(), [](MyRans &r) { r.reset(); });

            return result;
        }

    protected:
        struct Symbol {
            struct EncodedSymbol {
                CdfType start, frequency;
            };
            struct BypassSymbol {
                ValueType value;
                uint8_t bits;
            };

            enum {
                Encoded, Bypass
            } type;
            std::variant<EncodedSymbol, BypassSymbol> symbol;
        };

        std::array<MyRans, NInterleaved> rans;
        std::vector<RansBitstreamType> bitstream;
        std::vector<Symbol> symbolBuffer;

        void bufferSymbol(const ValueType value, const Cdf cdf) {
            auto startAndFrequency = cdf.getStartAndFrequency(value);
            if (startAndFrequency.has_value()) {
                auto [start, frequency] = startAndFrequency.value();
                symbolBuffer.push_back({Symbol::Encoded, typename Symbol::EncodedSymbol({start, frequency})});
            } else {
                // TODO: handle when value < 0

                uint8_t bits = sizeof(ValueType) * 8 - std::countl_zero(static_cast<unsigned int>(value));
                symbolBuffer.push_back({Symbol::Bypass, typename Symbol::BypassSymbol({value, bits})});
            }
        }

        void encodeSymbol(MyRans &encoder, const Symbol &symbol) {
            if (symbol.type == Symbol::Encoded) [[likely]] {
                const auto &encodedSymbol = std::get<typename Symbol::EncodedSymbol>(symbol.symbol);
                renorm(encoder, encodedSymbol.frequency);

                encoder.encPut(encodedSymbol.start, encodedSymbol.frequency);
            } else {
                // TODO: implement bypass coding
            }
        }

        void renorm(MyRans &encoder, const CdfType frequency) {
            if constexpr (MyRans::oneShotRenorm) {
                auto output = encoder.encRenormOnce(frequency);
                if (output.has_value()) bitstream.push_back(output.value());
            } else {
                auto output = encoder.encRenormOnce(frequency);
                while (output.has_value()) {
                    bitstream.push_back(output.value());
                    output = encoder.encRenormOnce(frequency);
                }
            }
        }
    };
}

#endif //RECOIL_RANS_ENCODER_H
