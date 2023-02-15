#ifndef RECOIL_RANS_H
#define RECOIL_RANS_H

#include "recoil/lib/cdf.h"
#include <cstdint>
#include <type_traits>
#include <optional>

namespace Recoil {
    using ValueType = int16_t;
    using BitCountType = uint8_t;

    template<class T>
    concept UnsignedType = std::is_unsigned_v<T>;

    template<UnsignedType RansStateType, UnsignedType RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits = 8 *
                                                                                            sizeof(RansBitstreamType)>
    class Rans {
        static_assert(WriteBits <= sizeof(RansBitstreamType) * 8,
                      "WriteBits cannot be greater than the size of RansBitstreamType");
        static_assert(ProbBits <= 16, "ProbBits must be smaller or equal to 16");
    public:
        RansStateType state;

        Rans() : state(RenormLowerBound) {}

        explicit Rans(RansStateType state) : state(state) {}

        static constexpr const bool oneShotRenorm = WriteBits >= ProbBits;

        inline void reset() { state = RenormLowerBound; }

        inline void encPut(CdfType start, CdfType frequency) {
            state = ((state / frequency) << ProbBits) + (state % frequency) + start;
        }

        inline void encPutBypass(ValueType value, BitCountType bits) {
            state = (state << bits) | value;
        }

        inline std::optional<RansBitstreamType> encRenormOnce(const CdfType frequency) {
            const RansStateType renormUpperBound = ((RenormLowerBound >> ProbBits) << WriteBits) * frequency;
            if (state >= renormUpperBound) {
                const RansBitstreamType mask = (1ul << WriteBits) - 1;
                Recoil::UnsignedType auto output = static_cast<RansBitstreamType>(state & mask);
                state >>= WriteBits;
                return output;
            } else return std::nullopt;
        }

        [[nodiscard]] inline CdfType decGetProbability() const {
            constexpr RansStateType mask = (1ul << ProbBits) - 1;
            return state & mask;
        }

        inline ValueType decGetBypass(const BitCountType nbits) {
            const RansStateType mask = (1ul << nbits) - 1;
            auto value = static_cast<ValueType>(state & mask);
            state >>= nbits; // FIXME: should be const function
            return value;
        }

        inline void decAdvanceSymbol(const CdfType lastStart, const CdfType lastFrequency) {
            constexpr RansStateType mask = (1ul << ProbBits) - 1;
            state = (state >> ProbBits) * lastFrequency + (state & mask) - lastStart;
        }

        inline bool decRenormOnce(const RansBitstreamType next) {
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
}

#endif //RECOIL_RANS_H
