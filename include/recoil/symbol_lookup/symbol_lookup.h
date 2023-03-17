#ifndef RECOIL_SYMBOL_LOOKUP_H
#define RECOIL_SYMBOL_LOOKUP_H

#include "recoil/lib/cuda/macros.h"
#include "recoil/symbol_lookup/lookup_mode.h"
#include "recoil/symbol_lookup/lut.h"
#include "recoil/symbol_lookup/cdf_lut_pool.h"

#include <concepts>
#include <cstdint>
#include <optional>

namespace Recoil {
    template <std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            uint8_t ProbBits, uint8_t LutGranularity>
    class SymbolLookup {
    protected:
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MyLutItem = LutItem<CdfType, ValueType, ProbBits, LutGranularity>;
    public:
        struct SymbolInfo {
            ValueType value;
            CdfType start, frequency;
        };

        /*
         * It is the caller's responsibility to ensure that the pool is not destructed when this class is used.
         * For performance reasons this will not be checked & the two pool pointers will be simply copied here.
         */
        explicit CUDA_HOST_DEVICE SymbolLookup(const MyCdfLutPool& pool) : cdfPool(pool.getCdfPool()), lutPool(pool.getLutPool()) {}

        [[nodiscard]] CUDA_HOST_DEVICE inline SymbolInfo getSymbolInfo(
                CdfLutOffsetType cdfOffset, CdfLutOffsetType lutOffset, CdfType probability) const {
            if constexpr (LutOnlyGranularity<LutGranularity>) {
                auto lutItem = getLut(lutOffset)[probability];
                return SymbolInfo{ lutItem.getValue(), lutItem.getStart(), lutItem.getFrequency() };
            }

            auto startOffset = 0;
            if constexpr (MixedGranularity<LutGranularity>) {
                startOffset = getLut(lutOffset)[probability >> (LutGranularity - 1)].getValue();
            }

            return linearSearch(cdfOffset, probability, startOffset);
        }

        [[nodiscard]] inline SymbolInfo getSymbolInfo(CdfLutOffsetType cdfOffset, ValueType symbol) const {
            auto cdf = getCdf(cdfOffset);
            return { symbol, cdf[symbol], static_cast<CdfType>(cdf[symbol + 1] - cdf[symbol]) };
        }

    protected:
        const CdfType * __restrict__ cdfPool;
        const MyLutItem * __restrict__ lutPool;

        [[nodiscard]] CUDA_HOST_DEVICE inline const CdfType * __restrict__ getCdf(CdfLutOffsetType cdfOffset) const { return cdfPool + cdfOffset; }
        [[nodiscard]] CUDA_HOST_DEVICE inline const MyLutItem * __restrict__ getLut(CdfLutOffsetType lutOffset) const { return lutPool + lutOffset; }

        [[nodiscard]] CUDA_HOST_DEVICE inline SymbolInfo linearSearch(
                CdfLutOffsetType cdfOffset, CdfType probability, CdfLutOffsetType startOffset = 0) const {
            auto cdf = getCdf(cdfOffset);
            for (auto *it = cdf + startOffset + 1; *it != 0; it++) {
                if (*it > probability) {
                    return SymbolInfo{ static_cast<ValueType>(it - 1 - cdf) , *(it - 1), static_cast<CdfType>(*it - *(it - 1)) };
                }
            }

            // TODO: handle the case when the value is not found; used for bypass coding
        }
    };
}

#endif //RECOIL_SYMBOL_LOOKUP_H
