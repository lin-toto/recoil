#ifndef RECOIL_CDF_H
#define RECOIL_CDF_H

#include "recoil/type_aliases.h"
#include <span>
#include <optional>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
    template <typename T>
    constexpr T mypow(T num, unsigned int pow) {
        return pow == 0 ? 1 : num * mypow(num, pow - 1);
    }
}

namespace Recoil {
    class Cdf {
        // TODO: make a CUDA-compatible CDF class
        // TODO: maybe make it a template class in the future?
    public:
        /*
         * BitKnit like mixed LUT/CDF lookup. In bits.
         * 0: no LUT.
         * 1: LUT covers all values.
         * n: LUT covers the first ProbBits - n bits of values.
         */
        static const unsigned int LutGranularity = 0;

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
            if constexpr (LutGranularity == 1) {
                return lut[probability];
            } else {
				auto offset = 0;
				if(LutGranularity == 0){
					
				} else {
					offset=lut[probability>>(LutGranularity-1)];
				}
				auto it = std::find_if(cdf.begin() + offset, cdf.end(), [probability](auto v) {
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
			std::array<ValueType, mypow(2, ProbBits)> resultTrim{};
            for (auto it = cdf.begin() + 1; it != cdf.end(); it++) {
                for (auto i = *(it - 1); i < *it; i++) {
                    result[i] = it - cdf.begin() - 1;
					//std::cout<<i<<" "<<result[i]<<std::endl;
                }
            }
			if(LutGranularity>1){
				//std::cout<<result.size()<<std::endl;
				for (int i = 0; i<(result.size()>>(LutGranularity-1)); i++){
					resultTrim[i]=result[i>>(LutGranularity-1)];
					//std::cout<<i<<" "<<resultTrim[i]<<std::endl;
					//std::cout<<(i>>(LutGranularity-1))<<std::endl;
				}
				return resultTrim;
			} else {
				return result;
			}
            
        }
    };
}

#endif //RECOIL_CDF_H
