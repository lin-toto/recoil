#ifndef RECOIL_RANS_H
#define RECOIL_RANS_H

#include <cstdint>
#include <type_traits>

template<typename RansStateType, typename RansOutputType,
        uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBitCount = 8 * sizeof(RansOutputType)>
class Rans {
    static_assert(std::is_unsigned_v<RansStateType>, "RansStateType must be an unsigned type");
    static_assert(std::is_unsigned_v<RansOutputType>, "RansStateType must be an unsigned type");
    static_assert(WriteBitCount <= sizeof(RansOutputType) * 8, "WriteBitCount cannot be greater than the size of RansOutputType");
public:
    Rans () : state(RenormLowerBound) {}
    explicit Rans (RansStateType state) : state(state) {}
protected:
    RansStateType state;

    static constexpr bool oneShotRenorm = WriteBitCount >= ProbBits;
};

template<uint8_t ProbBits, uint64_t RenormLowerBound = 1ull << 31>
using Rans64 = Rans<uint64_t, uint32_t, ProbBits, RenormLowerBound>;

template<uint8_t ProbBits, uint64_t RenormLowerBound = 1u << 16>
using Rans32 = Rans<uint32_t, uint16_t, ProbBits, RenormLowerBound>;

#endif //RECOIL_RANS_H
