#ifndef RECOIL_SYMBOL_LOOKUP_H
#define RECOIL_SYMBOL_LOOKUP_H

#include "recoil/cuda/macros.h"
#include "cdf_lut_pool.h"

#include <concepts>
#include <cstdint>
#include <optional>

namespace Recoil {
    template <std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            uint8_t ProbBits, uint8_t LutGranularity>
    class SymbolLookup {
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
    public:
        struct StartAndFrequency {
            CdfType start, frequency;
        };

        /*
         * It is the caller's responsibility to ensure that the pool is not destructed when this class is used.
         * For performance reasons this will not be checked & the two pool pointers will be simply copied here.
         */
        explicit SymbolLookup(const MyCdfLutPool& pool) : cdfPool(pool.getCdfPool()), lutPool(pool.getLutPool()) {}

        enum LookupMode { CdfOnly, LutOnly, Mixed };
        inline constexpr LookupMode lookupMode() const {
            if constexpr (LutGranularity == 1) return LutOnly;
            if constexpr (LutGranularity > 1) return Mixed;
            return CdfOnly; // LutGranularity == 0
        }

        [[nodiscard]] CUDA_HOST_DEVICE inline ValueType findValue(
                CdfLutOffsetType cdfOffset, CdfLutOffsetType lutOffset, CdfType probability) const {
            auto startOffset = 0;
            if constexpr (lookupMode() != CdfOnly)
                startOffset = getLut(lutOffset)[probability >> (LutGranularity - 1)];
            if constexpr (lookupMode() == LutOnly)
                return startOffset;

            auto cdf = getCdf(cdfOffset);
            for (auto *it = cdf + startOffset; *it != 0; it++) {
                if (*it > probability) return it - cdf - 1;
            }

            // TODO: handle the case when the value is not found; used for bypass coding
        }

        [[nodiscard]] CUDA_HOST_DEVICE inline StartAndFrequency getStartAndFrequency(
                CdfLutOffsetType cdfOffset, ValueType symbol) const {
            auto cdf = getCdf(cdfOffset);
            return { cdf[symbol], cdf[symbol + 1] - cdf[symbol] };
        }
    protected:
        CdfType *cdfPool;
        ValueType *lutPool;

        [[nodiscard]] inline CdfType *getCdf(CdfLutOffsetType cdfOffset) const { return cdfPool + cdfOffset; }
        [[nodiscard]] inline ValueType *getLut(CdfLutOffsetType lutOffset) const { return lutPool + lutOffset; }
    };
}

#endif //RECOIL_SYMBOL_LOOKUP_H
