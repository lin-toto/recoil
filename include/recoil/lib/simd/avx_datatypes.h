#ifndef RECOIL_AVX_DATATYPES_H
#define RECOIL_AVX_DATATYPES_H

#include <x86intrin.h>
#include <array>

namespace Recoil {
    template <typename T>
    concept SimdDataTypeWrapperConcept = requires(T) {
        { typename T::SimdDataType{} };
        { typename T::ArrayType{} };
    };

    using u32x8 = __m256i;
    struct u32x8_wrapper {
        using SimdDataType = u32x8;
        using ArrayType = std::array<uint32_t, 8>;

        [[nodiscard]] static inline u32x8 toSimd(const uint32_t *val) {
            return _mm256_load_si256(reinterpret_cast<const u32x8*>(val));
        }

        [[nodiscard]] static inline u32x8 toSimd(const ArrayType &val) {
            return toSimd(val.data());
        }

        [[nodiscard]] static inline ArrayType fromSimd(const u32x8 simd) {
            alignas(32) ArrayType val;
            _mm256_store_si256(reinterpret_cast<u32x8*>(val.begin()), simd);
            return val;
        }

        [[nodiscard]] static inline u32x8 setAll(const uint32_t val) {
            return _mm256_set1_epi32(val);
        }
    };

    using u32x16 = __m512i;
    struct u32x16_wrapper {
        using SimdDataType = u32x16;
        using ArrayType = std::array<uint32_t, 16>;

        [[nodiscard]] static inline u32x16 toSimd(const uint32_t *val) {
            return _mm512_load_si512(reinterpret_cast<const u32x16*>(val));
        }

        [[nodiscard]] static inline u32x16 toSimd(const ArrayType &val) {
            return toSimd(val.data());
        }

        [[nodiscard]] static inline ArrayType fromSimd(const u32x16 simd) {
            alignas(64) ArrayType val;
            _mm512_store_si512(reinterpret_cast<u32x8*>(val.begin()), simd);
            return val;
        }

        [[nodiscard]] static inline u32x16 setAll(const uint32_t val) {
            return _mm512_set1_epi32(val);
        }
    };
}

#endif //RECOIL_AVX_DATATYPES_H
