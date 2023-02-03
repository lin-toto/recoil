#ifndef RECOIL_CDF_H
#define RECOIL_CDF_H

#include <span>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace Recoil {
    using CdfType = uint16_t;

    class Cdf {
    public:
        const std::span<CdfType> cdf;

        explicit Cdf(std::span<CdfType> cdf) : cdf(cdf) {}

        [[nodiscard]] inline std::optional<std::pair<CdfType, CdfType>> getStartAndFrequency(int value) const {
            if (value >= 0 && value < cdf.size() - 1) [[likely]]
                return std::make_pair(cdf[value], cdf[value + 1] - cdf[value]);
            else
                return std::nullopt;
        }

        [[nodiscard]] inline std::optional<unsigned int> findValue(CdfType probability) const {
            auto it = std::find_if(cdf.begin(), cdf.end(), [probability](auto v) {
                return v > probability;
            });

            if (it != cdf.end()) [[likely]]
                return it - cdf.begin() - 1;
            else
                return std::nullopt;
        }
    };
}

#endif //RECOIL_CDF_H
