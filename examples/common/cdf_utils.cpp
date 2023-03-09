#include "cdf_utils.h"
#include <numeric>
#include <stdexcept>

namespace Recoil::Examples {
    std::vector<CdfType> buildCdfFromString(std::string_view str, uint8_t probBits) {
        static const auto characterCount = 256;
        const CdfType targetSum = (1 << probBits) - 1;

        std::array<size_t, characterCount> count{};
        for (unsigned char chr : str) count[chr]++;


        std::array<size_t, characterCount + 1> rawCdf{};
        std::partial_sum(count.begin(), count.end(), rawCdf.begin() + 1);

        for (auto &val : rawCdf) val = (targetSum * val) / rawCdf.back();

        std::vector<CdfType> cdf(rawCdf.begin(), rawCdf.end());
        fixCdfZerosByStealing(cdf);

        return cdf;
    }

    void fixCdfZerosByStealing(std::vector<CdfType> &cdf) {
        // Algorithm from ryg_rans.
        for (auto it = cdf.begin(); it != cdf.end() - 1; it++) {
            if (*(it + 1) == *it) {
                auto bestFreq = std::numeric_limits<CdfType>::max();
                auto bestSteal = cdf.end();
                for (auto j = cdf.begin(); j != cdf.end() - 1; j++) {
                    auto freq = *(j + 1) - *j;
                    if (freq > 1 && freq < bestFreq) {
                        bestFreq = freq;
                        bestSteal = j;
                    }
                }
                if (bestSteal == cdf.end()) [[unlikely]] throw std::runtime_error("Cannot steal freq");

                if (bestSteal < it)
                    std::transform(bestSteal + 1, it + 1, bestSteal + 1, [](auto v) { return v - 1; });
                else
                    std::transform(it + 1, bestSteal + 1, it + 1, [](auto v) { return v + 1; });
            }
        }
    }

    std::vector<ValueType> buildLut(std::span<CdfType> cdf, uint8_t probBits, uint8_t lutGranularity) {
        std::vector<ValueType> result;
        result.resize(1 << (probBits - lutGranularity + 1));

        for (auto it = cdf.begin() + 1; it != cdf.end(); it++) {
            std::fill(
                    result.begin() + ((*(it - 1) + (1 << (lutGranularity - 1)) - 1) >> (lutGranularity - 1)),
                    result.begin() + (*it >> (lutGranularity - 1)),
                    it - cdf.begin() - 1
            );
        }
        return result;
    }
}