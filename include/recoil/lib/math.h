#ifndef RECOIL_MATH_H
#define RECOIL_MATH_H

namespace Recoil {
    template<class T>
    constexpr T saveDiv(T a, T b) {
        return (a + b - 1) / b;
    }

    constexpr uint8_t floorlog2(const uint8_t x) {
        return x == 1 ? 0 : 1 + floorlog2(x >> 1);
    }

    constexpr uint8_t ceillog2(const uint8_t x) {
        return x == 1 ? 0 : floorlog2(x - 1) + 1;
    }
}

#endif //RECOIL_MATH_H
