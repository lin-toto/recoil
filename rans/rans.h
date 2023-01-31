#ifndef RECOIL_RANS_H
#define RECOIL_RANS_H

#include "lib/cdf.h"
#include <cstdint>
#include <type_traits>
#include <optional>

using ValueType = int16_t;
using BitCountType = uint8_t;

template<typename RansStateType, typename RansBitstreamType,
        BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits = 8 * sizeof(RansBitstreamType)>
class Rans {
    static_assert(std::is_unsigned_v<RansStateType>, "RansStateType must be an unsigned type");
    static_assert(std::is_unsigned_v<RansBitstreamType>, "RansStateType must be an unsigned type");
    static_assert(WriteBits <= sizeof(RansBitstreamType) * 8, "WriteBits cannot be greater than the size of RansBitstreamType");
    static_assert(ProbBits <= 16, "ProbBits must be smaller or equal to 16");
public:
    Rans () : state(RenormLowerBound) {}
    explicit Rans (RansStateType state) : state(state) {}

    static constexpr bool oneShotRenorm = WriteBits >= ProbBits;

    RansStateType state;

    void encPut(CdfType start, CdfType frequency) {
        state = ((state / frequency) << ProbBits) + (state % frequency) + start;
    }

    void encPutBypass(ValueType value, BitCountType bits) {
        state = (state << bits) | value;
    }

    std::optional<RansBitstreamType> encRenormOnce(CdfType frequency) {
        const RansStateType renormUpperBound = ((RenormLowerBound >> ProbBits) << WriteBits) * frequency;
        if (state > renormUpperBound) {
            const RansBitstreamType mask = (1 << WriteBits) - 1;
            auto output = static_cast<RansBitstreamType>(state & mask);
            state >>= WriteBits;
            return mask;
        } else return std::nullopt;
    }

    CdfType decGetCdf() {
        const RansStateType mask = (1 << ProbBits) - 1;
        return state & mask;
    }

    ValueType decGetBypass(BitCountType nbits) {
        RansStateType mask = (1 << nbits) - 1;
        auto value = static_cast<ValueType>(state & mask);
        state >>= nbits;
        return value;
    }

    void decAdvanceSymbol(CdfType lastStart, CdfType lastFrequency) {
        const RansStateType mask = (1 << ProbBits) - 1;
        state = (state >> ProbBits) * lastFrequency + (state & mask) + lastStart;
    }

    bool decRenormOnce(RansBitstreamType next) {
        if (state < RenormLowerBound) {
            state = (state << WriteBits) | next;
            return true;
        } else return false;
    }
};

template<BitCountType ProbBits, auto RenormLowerBound = 1ull << 31>
using Rans64 = Rans<uint64_t, uint32_t, ProbBits, RenormLowerBound>;

template<BitCountType ProbBits, auto RenormLowerBound = 1u << 16>
using Rans32 = Rans<uint32_t, uint16_t, ProbBits, RenormLowerBound>;

#endif //RECOIL_RANS_H
