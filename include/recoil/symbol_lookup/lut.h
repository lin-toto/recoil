#ifndef RECOIL_LUT_H
#define RECOIL_LUT_H

#include "lookup_mode.h"
#include "recoil/lib/cuda/macros.h"

#include <concepts>
#include <cstdint>
#include <span>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity>
    struct LutItem {
        LutItem(ValueType value, CdfType start, CdfType frequency) noexcept {}
    };

    template<uint8_t ProbBits, uint8_t LutGranularity> requires (LutOnlyGranularity<LutGranularity> && ProbBits <= 12)
    struct LutItem<uint16_t, uint8_t, ProbBits, LutGranularity> {
        // Special optimization for packed LUT. Value, start and frequency can be further packed into a 32-bit integer.
        uint32_t packed;

        LutItem(uint8_t value, uint16_t start, uint16_t frequency) noexcept {
            packed = value | (start << 8) | (frequency << 20);
        }

        [[nodiscard]] CUDA_HOST_DEVICE inline uint8_t getValue() const { return packed & 0xff; };
        [[nodiscard]] CUDA_HOST_DEVICE inline uint16_t getStart() const { return (packed >> 8) & 0x0fff; };
        [[nodiscard]] CUDA_HOST_DEVICE inline uint16_t getFrequency() const { return (packed >> 20) & 0x0fff; };
    };

    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity>
    requires LutOnlyGranularity<LutGranularity>
    struct alignas(sizeof(uint64_t)) LutItem<CdfType, ValueType, ProbBits, LutGranularity> {
        // Generic packed LUT implementation, so that only one memory lookup is necessary.
        // AVX2 implementation requires CdfType -> uint16_t, and they come first, so that padding is easier to handle.

        CdfType start, frequency;
        ValueType value;

        LutItem(ValueType value, CdfType start, CdfType frequency) noexcept
            : value(value), start(start), frequency(frequency) {}

        [[nodiscard]] CUDA_HOST_DEVICE inline ValueType getValue() const { return value; };
        [[nodiscard]] CUDA_HOST_DEVICE inline CdfType getStart() const { return start; };
        [[nodiscard]] CUDA_HOST_DEVICE inline CdfType getFrequency() const { return frequency; };
    }; // Required to align to 64 bits for AVX2 implementation

    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity>
    requires MixedGranularity<LutGranularity>
    struct LutItem<CdfType, ValueType, ProbBits, LutGranularity> {
        ValueType value;

        LutItem(ValueType value, CdfType start, CdfType frequency) noexcept
            : value(value) {}

        [[nodiscard]] CUDA_HOST_DEVICE inline ValueType getValue() const { return value; };
    };

    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType, uint8_t ProbBits, uint8_t LutGranularity>
    class LutBuilder {
        using MyLutItem = LutItem<CdfType, ValueType, ProbBits, LutGranularity>;
    public:
        static std::vector<MyLutItem> buildLut(std::span<CdfType> cdf) {
            std::vector<MyLutItem> result;
            if constexpr (CdfOnlyGranularity<LutGranularity>)
                return result;
            else {
                result.resize(1 << (ProbBits - LutGranularity + 1), { 0, 0, 0 });

                for (auto it = cdf.begin() + 1; it <= cdf.end(); it++) {
                    auto prob = it == cdf.end() ? (1 << ProbBits) : *it;
                    std::fill(
                            result.begin() + ((*(it - 1) + (1 << (LutGranularity - 1)) - 1) >> (LutGranularity - 1)),
                            result.begin() + (prob >> (LutGranularity - 1)),
                            MyLutItem(it - cdf.begin() - 1, *(it - 1), static_cast<CdfType>(prob - *(it - 1)))
                    );
                }
                return result;
            }
        }
    };

    template<uint8_t ProbBits, uint8_t LutGranularity>
    using LutBuilder_8bitSymbol = LutBuilder<uint16_t, uint8_t, ProbBits, LutGranularity>;
    template<uint8_t ProbBits, uint8_t LutGranularity>
    using LutBuilder_16bitSymbol = LutBuilder<uint16_t, uint16_t, ProbBits, LutGranularity>;
}

#endif //RECOIL_LUT_H
