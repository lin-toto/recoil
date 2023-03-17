#ifndef RECOIL_SYMBOL_LOOKUP_AVX_BASE_H
#define RECOIL_SYMBOL_LOOKUP_AVX_BASE_H

#include "recoil/symbol_lookup/symbol_lookup.h"
#include "recoil/lib/simd/avx_datatypes.h"

namespace Recoil {
    template<std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity, SimdDataTypeWrapperConcept SimdDataTypeWrapper>
    class SymbolLookup_AVX_Base : public SymbolLookup<uint16_t, ValueType, ProbBits, LutGranularity> {
        using MyBase = SymbolLookup<uint16_t, ValueType, ProbBits, LutGranularity>;
        using SimdDataType = typename SimdDataTypeWrapper::SimdDataType;
    public:
        using MyBase::MyBase;

        struct SymbolInfo {
            SimdDataType value, start, frequency;
        };

        inline SymbolInfo getSymbolInfo(SimdDataType cdfOffsets, SimdDataType lutOffsets, SimdDataType probabilities) const {
            if constexpr (LutOnlyGranularity<LutGranularity>) {
                if constexpr (requires () { MyBase::MyLutItem::packed; }) {
                    return getSymbolInfo_lutOnly_packed(lutOffsets, probabilities);
                } else {
                    return getSymbolInfo_lutOnly(lutOffsets, probabilities);
                }
            } else {
                SimdDataType startPositions = SimdDataTypeWrapper::setAll(0);
                if constexpr (MixedGranularity<LutGranularity>) {
                    startPositions = valueOnlyLutLookup(lutOffsets, probabilities);
                }
                return getSymbolInfo_mixed(cdfOffsets, startPositions, probabilities);
            }
        }

    protected:
        virtual SimdDataType valueOnlyLutLookup(SimdDataType lutOffsets, SimdDataType probabilities) const = 0;
        virtual SymbolInfo getSymbolInfo_mixed(SimdDataType cdfOffsets, SimdDataType startPositions, SimdDataType probabilities) const = 0;
        virtual SymbolInfo getSymbolInfo_lutOnly(SimdDataType lutOffsets, SimdDataType probabilities) const = 0;
        virtual SymbolInfo getSymbolInfo_lutOnly_packed(SimdDataType lutOffsets, SimdDataType probabilities) const = 0;
    };
}

#endif //RECOIL_SYMBOL_LOOKUP_AVX_BASE_H
