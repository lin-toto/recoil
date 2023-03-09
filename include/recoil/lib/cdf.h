#ifndef RECOIL_CDF_H
#define RECOIL_CDF_H

#include "recoil/type_defs.h"
#include <span>
#include <optional>
#include <algorithm>

namespace Recoil {
     class [[deprecated]]  Cdf {
        // TODO: make a CUDA-compatible CDF class
        // TODO: maybe make it a template class in the future?
    public:
        /*
         * BitKnit like mixed LUT/CDF lookup. In bits.
         * 0: no LUT.
         * 1: LUT covers all values.
         * n: LUT covers the first ProbBits - n + 1 bits of values.
         */
        static const unsigned int LutGranularity = 1;

        std::span<CdfType> cdf;
        std::span<ValueType> lut;

        Cdf() : cdf() {}
        explicit Cdf(std::span<CdfType> cdf) : cdf(cdf), lut() {} // Should not be used when LutGranularity != 0
        Cdf(std::span<CdfType> cdf, std::span<ValueType> lut) : cdf(cdf), lut(lut) {}

        // Value may not be a valid value in cdf, so use int instead of ValueType
        [[nodiscard]] inline std::optional<std::pair<CdfType, CdfType>> getStartAndFrequency(int value) const {
            if (value >= 0 && value < cdf.size() - 1) [[likely]]
                return std::make_pair(cdf[value], cdf[value + 1] - cdf[value]);
            else
                return std::nullopt;
        }

        [[nodiscard]] inline std::optional<ValueType> findValue(CdfType probability) const {
            auto offset = 0;
            if constexpr (LutGranularity != 0) offset = lut[probability >> (LutGranularity - 1)];
            if constexpr (LutGranularity == 1) return offset;

            auto it = std::find_if(cdf.begin() + offset, cdf.end(), [probability](auto v) {
                return v > probability;
            });

            if (it != cdf.end()) [[likely]]
                return it - cdf.begin() - 1;
            else
                return std::nullopt;
        }

        template<uint8_t ProbBits>
        static auto buildLut(std::span<CdfType> cdf) {
            std::array<ValueType, (1 << (ProbBits - LutGranularity + 1))> result{};

            for (auto it = cdf.begin() + 1; it != cdf.end(); it++) {
                std::fill(
                        result.begin() + ((*(it - 1) + (1 << (LutGranularity - 1)) - 1) >> (LutGranularity - 1)),
                        result.begin() + (*it >> (LutGranularity - 1)),
                        it - cdf.begin() - 1
                );
            }
            return result;
        }
    };
}

#endif //RECOIL_CDF_H
