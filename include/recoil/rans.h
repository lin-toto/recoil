#ifndef RECOIL_RANS_H
#define RECOIL_RANS_H

#include "recoil/lib/cuda/macros.h"
#include <optional>
#include <concepts>
#include <cstdint>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits = 8 * sizeof(RansBitstreamType)>
    class Rans {
        static_assert(WriteBits <= sizeof(RansBitstreamType) * 8,
                      "WriteBits cannot be greater than the size of RansBitstreamType");
        static_assert(ProbBits <= 16, "ProbBits must be smaller or equal to 16");
    public:
        RansStateType state;

        Rans() : state(RenormLowerBound) {}

        explicit Rans(RansStateType state) : state(state) {}

        static constexpr const bool oneShotRenorm = WriteBits >= ProbBits;

        CUDA_HOST_DEVICE inline void reset() { state = RenormLowerBound; }

        CUDA_HOST_DEVICE inline void encPut(CdfType start, CdfType frequency) {
            state = ((state / frequency) << ProbBits) + (state % frequency) + start;
        }

        CUDA_HOST_DEVICE inline void encPutBypass(ValueType value, uint8_t bits) {
            state = (state << bits) | value;
        }

        CUDA_HOST_DEVICE inline bool encShouldRenorm(const CdfType frequency) {
            const RansStateType renormUpperBound = ((RenormLowerBound >> ProbBits) << WriteBits) * frequency;
            return state >= renormUpperBound;
        }

        CUDA_HOST_DEVICE inline RansBitstreamType encRenormOnce() {
            const RansBitstreamType mask = (1ul << WriteBits) - 1;

            std::unsigned_integral auto output = static_cast<RansBitstreamType>(state & mask);
            state >>= WriteBits;
            return output;
        }

        CUDA_HOST_DEVICE [[nodiscard]] inline CdfType decGetProbability() const {
            constexpr RansStateType mask = (1ul << ProbBits) - 1;
            return state & mask;
        }

        CUDA_HOST_DEVICE inline ValueType decGetBypass(const uint8_t nbits) {
            const RansStateType mask = (1ul << nbits) - 1;
            auto value = static_cast<ValueType>(state & mask);
            state >>= nbits; // FIXME: should be const function
            return value;
        }

        CUDA_HOST_DEVICE inline void decAdvanceSymbol(const CdfType lastStart, const CdfType lastFrequency) {
            constexpr RansStateType mask = (1ul << ProbBits) - 1;
            state = (state >> ProbBits) * lastFrequency + (state & mask) - lastStart;
        }

        CUDA_HOST_DEVICE inline bool decShouldRenorm() { return state < RenormLowerBound; }

        CUDA_HOST_DEVICE inline void decRenormOnce(const RansBitstreamType next) { state = (state << WriteBits) | next; }
    };

    template<std::unsigned_integral ValueType, uint8_t ProbBits, auto RenormLowerBound = 1ull << 31>
    using Rans64 = Rans<uint16_t, ValueType, uint64_t, uint32_t, ProbBits, RenormLowerBound>;

    template<std::unsigned_integral ValueType, uint8_t ProbBits, auto RenormLowerBound = 1u << 16>
    using Rans32 = Rans<uint16_t, ValueType, uint32_t, uint16_t, ProbBits, RenormLowerBound>;
}

#endif //RECOIL_RANS_H
