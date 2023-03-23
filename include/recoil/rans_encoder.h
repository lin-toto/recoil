#ifndef RECOIL_RANS_ENCODER_H
#define RECOIL_RANS_ENCODER_H

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/symbol_lookup/symbol_lookup.h"
#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"
#include <vector>
#include <span>
#include <bit>
#include <string>
#include <variant>
#include <cassert>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, uint8_t LutGranularity,
            size_t NInterleaved, bool RecordIntermediateStates = false>
    class RansEncoder {
        template<std::unsigned_integral, std::unsigned_integral, std::unsigned_integral T, std::unsigned_integral, uint8_t, T, uint8_t, uint8_t, size_t>
        friend class RansSplitEncoder;
        template<std::unsigned_integral, std::unsigned_integral, std::unsigned_integral T, std::unsigned_integral, uint8_t, T, uint8_t, uint8_t, size_t>
        friend class RansSymbolSplitEncoder;

        const size_t BitstreamLeftPadding = 16;
    protected:
        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MySymbolLookup = SymbolLookup<CdfType, ValueType, ProbBits, LutGranularity>;
        using MyRansCodedData = RansCodedData<
                CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>;
    public:
        RansEncoder(std::array<MyRans, NInterleaved> rans, const MyCdfLutPool& pool)
            : rans(std::move(rans)), symbolLookup(pool) {}

        /*
         * Buffer the values for encode with single shared CDF.
         */
        void buffer(const std::span<ValueType> values, const CdfLutOffsetType cdfOffset) {
            symbolBuffer.reserve(symbolBuffer.size() + values.size());

            for (const auto &value: values) {
                bufferSymbol(value, cdfOffset);
            }
        }

        /*
         * Buffer the values for encode with independent CDF for each symbol.
         */
        void buffer(const std::span<ValueType> values, const std::span<CdfLutOffsetType> cdfOffsets) {
            assert(values.size() == cdfOffsets.size());
            symbolBuffer.reserve(symbolBuffer.size() + values.size());

            for (auto i = 0; i < values.size(); i++) {
                bufferSymbol(values[i], cdfOffsets[i]);
            }
        }

        MyRansCodedData flush() {
            // AVX decoder requires 16 bytes of left padding
            bitstream.resize(BitstreamLeftPadding);
            encodeAll();

            MyRansCodedData result{symbolBuffer.size(), std::move(bitstream), BitstreamLeftPadding, std::move(rans)};
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
            RansStateType intermediateState;
            size_t symbolId;

            [[nodiscard]] inline size_t symbolGroupId () const { return symbolId / NInterleaved; }
            [[nodiscard]] inline size_t encoderId () const { return symbolId % NInterleaved; }
        };

        std::array<MyRans, NInterleaved> rans;
        MySymbolLookup symbolLookup;
        std::vector<RansBitstreamType> bitstream;
        std::vector<Symbol> symbolBuffer;
        std::vector<EncoderIntermediateState> intermediateStates;
        size_t symbolCounter = 0;

        inline void bufferSymbol(const ValueType value, const CdfLutOffsetType cdfId) {
            auto [_, start, frequency] = symbolLookup.getSymbolInfo(cdfId, value);

            symbolBuffer.push_back({symbolCounter, Symbol::Encoded, typename Symbol::EncodedSymbol({start, frequency})});
            symbolCounter++;

            /* TODO: support bypass coding
            uint8_t bits = sizeof(ValueType) * 8 - std::countl_zero(static_cast<unsigned int>(value));
            symbolBuffer.push_back({symbolCounter, Symbol::Bypass, typename Symbol::BypassSymbol({value, bits})}); */
        }

        /*
         * Encode the buffered values into the bitstream.
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
        void encodeAll() {
            auto ransIt = rans.rbegin();
            if constexpr (NInterleaved > 1) ransIt += NInterleaved - (symbolBuffer.size() - 1) % NInterleaved - 1;

            for (auto symbol = symbolBuffer.rbegin(); symbol != symbolBuffer.rend(); symbol++) {
                encodeSymbol(*ransIt, *symbol);

                if constexpr (NInterleaved > 1) {
                    ransIt++;
                    if (ransIt == rans.rend()) ransIt = rans.rbegin();
                }
            }
        }

        inline void encodeSymbol(MyRans &encoder, const Symbol &symbol) {
            if (symbol.type == Symbol::Encoded) [[likely]] {
                const auto &encodedSymbol = std::get<typename Symbol::EncodedSymbol>(symbol.symbol);
                bool renormed = renorm(encoder, encodedSymbol.frequency);
                if constexpr (RecordIntermediateStates) {
                    if (renormed) {
                        intermediateStates.push_back({encoder.state, symbol.symbolId});
                    }
                }

                encoder.encPut(encodedSymbol.start, encodedSymbol.frequency);
            } else {
                // TODO: implement bypass coding
                throw std::runtime_error("Bypass encoding not implemented");
            }
        }

        /*
         * Returns boolean representing if renormalization has occured.
         */
        inline bool renorm(MyRans &encoder, const CdfType frequency) {
            if (encoder.encShouldRenorm(frequency)) {
                auto output = encoder.encRenormOnce();
                if constexpr (MyRans::oneShotRenorm) {
                    bitstream.push_back(output);
                } else {
                    do {
                        bitstream.push_back(output);
                        output = encoder.encRenormOnce();
                    } while (encoder.encShouldRenorm(frequency));
                }
                return true;
            } else return false;
        }
    };
}

#endif //RECOIL_RANS_ENCODER_H
