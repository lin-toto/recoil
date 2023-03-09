#ifndef RECOIL_CDF_LUT_POOL_H
#define RECOIL_CDF_LUT_POOL_H

#include "recoil/cuda/macros.h"

#include <concepts>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>

namespace Recoil {
    using CdfLutOffsetType = uint32_t;

    /*
     * Memory Pool for CDF/LUT.
     * In this design value must be unsigned and starts from 0.
     *
     * LutGranularity: 0: no LUTs. 1: LUTs cover all values. n: LUTs cover the first ProbBits - n + 1 bits of values.
     */
    template <std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            uint8_t ProbBits, uint8_t LutGranularity>
    class CdfLutPool {
        static_assert(LutGranularity < ProbBits, "LutGranularity must be smaller than ProbBits");
    public:
        CdfLutPool(size_t cdfSize, size_t lutSize) : cdfSize(cdfSize), lutSize(lutSize) {
            if (cdfSize * sizeof(CdfType) > std::numeric_limits<CdfLutOffsetType>::max()
                || lutSize * sizeof(ValueType) > std::numeric_limits<CdfLutOffsetType>::max()) [[unlikely]]
                throw std::runtime_error("Allocation size is too big");

            const size_t size = cdfSize * sizeof(CdfType) + lutSize * sizeof(ValueType);
            pool = new uint8_t[size];
            std::fill(pool, pool + size, 0);

            cdfPool = reinterpret_cast<CdfType*>(pool);
            lutPool = reinterpret_cast<ValueType*>(pool + cdfSize * sizeof(CdfType));
        }

        CdfLutPool(CdfType *cdfPool, size_t cdfSize, ValueType *lutPool, size_t lutSize)
            : cdfPool(cdfPool), cdfSize(cdfSize), lutPool(lutPool), lutSize(lutSize) {}

        explicit CdfLutPool(const CdfLutPool&) = delete;
        CdfLutPool& operator=(const CdfLutPool&) = delete;

        ~CdfLutPool() {
            if (pool != nullptr) delete[] pool;
        }

        CUDA_HOST_DEVICE inline const CdfType *getCdfPool() const { return cdfPool; }
        CUDA_HOST_DEVICE inline const ValueType *getLutPool() const { return lutPool; }

        inline constexpr size_t eachLutSize() const {
            if constexpr (LutGranularity == 0) return 0; else return 1 << (ProbBits - LutGranularity + 1);
        }

        CdfLutOffsetType insertCdf(const std::vector<CdfType> &cdf) {
            // Ensure first position is 0, as it will be used as sentinel during lookup
            if (cdf[0] != 0) [[unlikely]] throw std::runtime_error("Invalid CDF; must start with 0");
            return insert(cdfPool, cdfPoolEnd, cdfSize, cdf);
        }

        CdfLutOffsetType insertLut(const std::vector<ValueType> &lut) {
            // Ensure the LUT length matches the assumption.
            if (lut.size() != eachLutSize()) [[unlikely]] throw std::runtime_error("Invalid LUT; must match size assumption");
            return insert(lutPool, lutPoolEnd, lutSize, lut);
        }
    protected:
        uint8_t *pool = nullptr;
        CdfType *cdfPool;
        ValueType *lutPool;
        CdfLutOffsetType cdfSize, lutSize;
        CdfLutOffsetType cdfPoolEnd = 0, lutPoolEnd = 0;

        template<typename DataType>
        inline CdfLutOffsetType insert(DataType *pool, CdfLutOffsetType &end, CdfLutOffsetType maxSize, const std::vector<DataType> &data) {
            if (end + data.size() > maxSize) [[unlikely]] {
                throw new std::runtime_error("Pool will overflow after insertion");
            }

            auto offset = end;
            std::copy(data.begin(), data.end(), pool + end);
            end += data.size();

            return offset;
        }
    };
}

#endif //RECOIL_CDF_LUT_POOL_H
