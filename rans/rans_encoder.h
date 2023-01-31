#ifndef RECOIL_RANS_ENCODER_H
#define RECOIL_RANS_ENCODER_H

#include "lib/cdf.h"
#include "rans/rans.h"
#include <vector>
#include <string>
#include <variant>
#include <cassert>

template<typename RansStateType, typename RansBitstreamType,
        BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits>
class RansEncoder {
    using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
public:
    explicit RansEncoder(MyRans rans) : rans(rans) {}

    // Buffer the values for encode with single shared CDF
    void encode(const std::vector<ValueType>& values, Cdf cdf) {
        symbolBuffer.reserve(values.size());

        for (const auto& value: values) {
            bufferSymbol(value, cdf);
        }
    }

    // Buffer the values for encode with independent CDF for each symbol
    void encode(const std::vector<ValueType>& values, const std::vector<Cdf>& cdfs) {
        assert(values.size() == cdfs.size());
        symbolBuffer.reserve(values.size());

        for (int i = 0; i < values.size(); i++) {
            bufferSymbol(values[i], cdfs[i]);
        }
    }

    std::string flush() {

    }
protected:
    struct Symbol {
        struct EncodedSymbol { CdfType start, frequency; };
        struct BypassSymbol { ValueType value; uint8_t bits; };

        enum { Encoded, Bypass } type;
        std::variant<EncodedSymbol, BypassSymbol> symbol;
    };

    MyRans rans;
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
};

#endif //RECOIL_RANS_ENCODER_H
