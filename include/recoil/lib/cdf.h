#ifndef RECOIL_CDF_H
#define RECOIL_CDF_H

#include "recoil/type_aliases.h"
#include <span>
#include <optional>
#include <algorithm>
#include <cmath>

namespace {
    template <typename T>
    constexpr T mypow(T num, unsigned int pow) {
        return pow == 0 ? 1 : num * mypow(num, pow - 1);
    }

    // TODO: simple hack here to test linear search and Lut
    const bool UseLut = true;
}

namespace Recoil {
    class Cdf {
    public:
        std::span<CdfType> cdf;
        std::span<ValueType> lut;

        Cdf() : cdf() {}
        explicit Cdf(std::span<CdfType> cdf) : cdf(cdf), lut() {}
        Cdf(std::span<CdfType> cdf, std::span<ValueType> lut) : cdf(cdf), lut(lut) {}

        // Value may not be a valid value in cdf, so use int instead of ValueType
        [[nodiscard]] inline std::optional<std::pair<CdfType, CdfType>> getStartAndFrequency(int value) const {
            if (value >= 0 && value < cdf.size() - 1) [[likely]]
                return std::make_pair(cdf[value], cdf[value + 1] - cdf[value]);
            else
                return std::nullopt;
        }

        [[nodiscard]] inline std::optional<ValueType> findValue(CdfType probability) const {
            if constexpr (UseLut) {
                return lut[probability];
            } else {
                auto it = std::find_if(cdf.begin(), cdf.end(), [probability](auto v) {
                    return v > probability;
                });

                if (it != cdf.end()) [[likely]]
                    return it - cdf.begin() - 1;
                else
                    return std::nullopt;
            }
        }

        template<uint8_t ProbBits>
        static auto buildLut(std::span<CdfType> cdf) {
            std::array<ValueType, mypow(2, ProbBits)> result{};

            for (auto it = cdf.begin() + 1; it != cdf.end(); it++) {
                for (auto i = *(it - 1); i < *it; i++) {
                    result[i] = it - cdf.begin() - 1;
                }
            }

            return result;
        }
    };
}

#endif //RECOIL_CDF_H
