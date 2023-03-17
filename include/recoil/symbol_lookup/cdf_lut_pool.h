#ifndef RECOIL_CDF_LUT_POOL_H
#define RECOIL_CDF_LUT_POOL_H

#include "recoil/lib/cuda/macros.h"
#include "recoil/symbol_lookup/lut.h"

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
        using MyCdfLutPool = CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity>;
        using MyLutItem = LutItem<CdfType, ValueType, ProbBits, LutGranularity>;
    //public:
        CdfLutPool(size_t cdfSize, size_t lutSize) : cdfSize(cdfSize), lutSize(lutSize) {
            if (cdfSize * sizeof(CdfType) > std::numeric_limits<CdfLutOffsetType>::max()
                || lutSize * sizeof(MyLutItem) > std::numeric_limits<CdfLutOffsetType>::max()) [[unlikely]]
                throw std::runtime_error("Allocation size is too big");

            const size_t size = poolSize();
            pool = new uint8_t[size];
            std::fill(pool, pool + size, 0);

            lutPool = reinterpret_cast<MyLutItem*>(pool);
            cdfPool = reinterpret_cast<CdfType*>(pool + lutSize * sizeof(MyLutItem));
        }

        CdfLutPool(const CdfType *cdfPool, size_t cdfSize, const MyLutItem *lutPool, size_t lutSize) noexcept
            : cdfPool(cdfPool), cdfSize(cdfSize), lutPool(lutPool), lutSize(lutSize), readOnly(true) {}

        explicit CdfLutPool(const CdfLutPool&) = delete;
        CdfLutPool& operator=(const CdfLutPool&) = delete;

        CdfLutPool(CdfLutPool&& other) noexcept : pool(std::exchange(other.pool, nullptr)),
            cdfPool(other.cdfPool), lutPool(other.lutPool), readOnly(std::exchange(other.readOnly, true)),
            cdfSize(other.cdfSize), lutSize(other.lutSize), cdfPoolEnd(other.cdfPoolEnd), lutPoolEnd(other.lutPoolEnd) {}

        ~CdfLutPool() {
            delete[] pool;
        }

        CUDA_HOST_DEVICE inline const uint8_t *getPool() const { return pool; }
        CUDA_HOST_DEVICE inline const CdfType *getCdfPool() const { return cdfPool; }
        CUDA_HOST_DEVICE inline const MyLutItem *getLutPool() const { return lutPool; }
        CUDA_HOST_DEVICE inline CdfLutOffsetType getCdfSize() const { return cdfSize; }
        CUDA_HOST_DEVICE inline CdfLutOffsetType getLutSize() const { return lutSize; }

        inline size_t poolSize() const { return cdfSize * sizeof(CdfType) + lutSize * sizeof(MyLutItem); }
        inline constexpr size_t eachLutSize() const {
            if constexpr (CdfOnlyGranularity<LutGranularity>) return 0;
            else return 1 << (ProbBits - LutGranularity + 1);
        }

        CdfLutOffsetType insertCdf(const std::vector<CdfType> &cdf) {
            // Ensure first position is 0, as it will be used as sentinel during lookup
            if (cdf[0] != 0) [[unlikely]] throw std::runtime_error("Invalid CDF; must start with 0");
            return insert(const_cast<CdfType*>(cdfPool), cdfPoolEnd, cdfSize, cdf);
        }

        CdfLutOffsetType insertLut(const std::vector<MyLutItem> &lut) {
            // Ensure the LUT length matches the assumption.
            if (lut.size() != eachLutSize()) [[unlikely]] throw std::runtime_error("Invalid LUT; must match size assumption");
            return insert(const_cast<MyLutItem*>(lutPool), lutPoolEnd, lutSize, lut);
        }

        inline void setReadOnly() { readOnly = true; }

        MyCdfLutPool createRef() noexcept {
            setReadOnly();
            return CdfLutPool(cdfPool, cdfSize, lutPool, lutSize);
        }
    protected:
        uint8_t *pool = nullptr;
        const CdfType *cdfPool;
        const MyLutItem *lutPool;

        bool readOnly = false;
        CdfLutOffsetType cdfSize, lutSize;
        CdfLutOffsetType cdfPoolEnd = 0, lutPoolEnd = 0;

        template<typename DataType>
        inline CdfLutOffsetType insert(DataType *pool, CdfLutOffsetType &end, CdfLutOffsetType maxSize, const std::vector<DataType> &data) {
            if (end + data.size() > maxSize) [[unlikely]] {
                throw std::runtime_error("Pool will overflow after insertion");
            }

            if (readOnly) [[unlikely]] {
                throw std::runtime_error("Readonly pool");
            }

            auto offset = end;
            std::copy(data.begin(), data.end(), pool + end);
            end += data.size();

            return offset;
        }
    };

    template<uint8_t ProbBits, uint8_t LutGranularity>
    using CdfLutPool_8bitSymbol = CdfLutPool<uint16_t, uint8_t, ProbBits, LutGranularity>;

    template<uint8_t ProbBits, uint8_t LutGranularity>
    using CdfLutPool_16bitSymbol = CdfLutPool<uint16_t, uint16_t, ProbBits, LutGranularity>;
}

#endif //RECOIL_CDF_LUT_POOL_H
