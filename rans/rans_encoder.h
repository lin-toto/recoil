#ifndef RECOIL_RANS_ENCODER_H
#define RECOIL_RANS_ENCODER_H

#include "lib/cdf.h"
#include "rans/rans.h"
#include "rans/rans_coded_data.h"
#include <vector>
#include <ranges>
#include <string>
#include <variant>
#include <cassert>

template<typename RansStateType, typename RansBitstreamType,
        BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
        size_t nInterleaved>
class RansEncoder {
    using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
    using MyRansCodedData = RansCodedData<
            RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, nInterleaved>;
public:
    explicit RansEncoder(std::array<MyRans, nInterleaved> rans) : rans(std::move(rans)) {}

    // Buffer the values for encode with single shared CDF
    void buffer(const std::vector<ValueType>& values, Cdf cdf) {
        symbolBuffer.reserve(symbolBuffer.size() + values.size());

        for (const auto& value: values) {
            bufferSymbol(value, cdf);
        }
    }

    // Buffer the values for encode with independent CDF for each symbol
    void buffer(const std::vector<ValueType>& values, const std::vector<Cdf>& cdfs) {
        assert(values.size() == cdfs.size());
        symbolBuffer.reserve(symbolBuffer.size() + values.size());

        for (auto i = 0; i < values.size(); i++) {
            bufferSymbol(values[i], cdfs[i]);
        }
    }

    MyRansCodedData flush() {
        auto ransIt = rans.begin();
        for (auto& symbol : std::ranges::reverse_view(symbolBuffer)) {
            encodeSymbol(ransIt, symbol);

            ransIt++;
            if (ransIt == rans.end()) ransIt = rans.begin();
        }

        MyRansCodedData result{ bitstream, rans };

        bitstream.clear();
        symbolBuffer.clear();
        std::for_each(rans.begin(), rans.end(), [](MyRans& r) { r.reset(); });

        return result;
    }
protected:
    struct Symbol {
        struct EncodedSymbol { CdfType start, frequency; };
        struct BypassSymbol { ValueType value; uint8_t bits; };

        enum { Encoded, Bypass } type;
        std::variant<EncodedSymbol, BypassSymbol> symbol;
    };

    std::array<MyRans, nInterleaved> rans;
    std::vector<RansBitstreamType> bitstream;
    std::vector<Symbol> symbolBuffer;

    void bufferSymbol(ValueType value, Cdf cdf) {
        auto startAndFrequency = cdf.getStartAndFrequency(value);
        if (startAndFrequency.has_value()) {
            auto [start, frequency] = startAndFrequency.value();
            symbolBuffer.push_back({ Symbol::Encoded, { start, frequency } });
        } else {
            uint8_t bits = sizeof(ValueType) * 8 - __builtin_clz(value);
            symbolBuffer.push_back({ Symbol::Bypass, { value, bits } });
        }
    }

    void encodeSymbol(MyRans& encoder, const Symbol& symbol) {
        if (symbol.type == Symbol::Encoded) [[likely]] {
            const auto& encodedSymbol = std::get<Symbol::EncodedSymbol>(symbol.symbol);
            renorm(encoder, encodedSymbol.frequency);

            encoder.encPut(encodedSymbol.start, encodedSymbol.frequency);
        } else {
            // TODO: implement bypass coding
        }
    }

    void renorm(MyRans& encoder, CdfType frequency) {
        if constexpr (encoder.oneShotRenorm) {
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

#endif //RECOIL_RANS_ENCODER_H
