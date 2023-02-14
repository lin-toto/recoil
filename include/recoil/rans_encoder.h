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
    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t NInterleaved,
            bool RecordIntermediateStates = false, UnsignedType RansIntermediateStateType = RansStateType>
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
            reset();

            return result;
        }

        void reset() {
            bitstream.clear();
            symbolBuffer.clear();
            if constexpr (RecordIntermediateStates) intermediateStates.clear();
            std::for_each(rans.begin(), rans.end(), [](MyRans &r) { r.reset(); });
            symbolCounter = 0;
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

            size_t symbolId;
            enum {
                Encoded, Bypass
            } type;
            std::variant<EncodedSymbol, BypassSymbol> symbol;
        };

        struct EncoderIntermediateState {
            RansIntermediateStateType intermediateState;
            size_t symbolId;

            [[nodiscard]] size_t encoderId () const { return symbolId % NInterleaved; }
        };

        std::array<MyRans, NInterleaved> rans;
        std::vector<RansBitstreamType> bitstream;
        std::vector<Symbol> symbolBuffer;
        std::vector<EncoderIntermediateState> intermediateStates;
        size_t symbolCounter = 0;

        inline void bufferSymbol(const ValueType value, const Cdf cdf) {
            auto startAndFrequency = cdf.getStartAndFrequency(value);
            if (startAndFrequency.has_value()) {
                auto [start, frequency] = startAndFrequency.value();
                symbolBuffer.push_back({symbolCounter, Symbol::Encoded, typename Symbol::EncodedSymbol({start, frequency})});
            } else {
                // TODO: handle when value < 0

                uint8_t bits = sizeof(ValueType) * 8 - std::countl_zero(static_cast<unsigned int>(value));
                symbolBuffer.push_back({symbolCounter, Symbol::Bypass, typename Symbol::BypassSymbol({value, bits})});
            }

            symbolCounter++;
        }

        inline void encodeSymbol(MyRans &encoder, const Symbol &symbol) {
            if (symbol.type == Symbol::Encoded) [[likely]] {
                const auto &encodedSymbol = std::get<typename Symbol::EncodedSymbol>(symbol.symbol);
                bool renormed = renorm(encoder, encodedSymbol.frequency);
                if constexpr (RecordIntermediateStates) {
                    if (renormed) {
                        intermediateStates.push_back({
                            static_cast<RansIntermediateStateType>(encoder.state),
                            symbol.symbolId
                        });
                    }
                }

                encoder.encPut(encodedSymbol.start, encodedSymbol.frequency);
            } else {
                // TODO: implement bypass coding
            }
        }

        /*
         * Returns boolean representing if renormalization has occured.
         */
        inline bool renorm(MyRans &encoder, const CdfType frequency) {
            auto output = encoder.encRenormOnce(frequency);
            if constexpr (MyRans::oneShotRenorm) {
                if (output.has_value()) {
                    bitstream.push_back(output.value());
                }
                return output.has_value();
            } else {
                bool renormFlag = false;
                while (output.has_value()) {
                    renormFlag = true;
                    bitstream.push_back(output.value());
                    output = encoder.encRenormOnce(frequency);
                }
                return renormFlag;
            }
        }
    };
}

#endif //RECOIL_RANS_ENCODER_H
