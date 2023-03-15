#ifndef RECOIL_RANS_CODED_DATA_CUDA_H
#define RECOIL_RANS_CODED_DATA_CUDA_H

#include "recoil/rans.h"
#include "recoil/rans_coded_data.h"

#include <concepts>
#include <cstdint>
#include <cuda/std/array>

namespace Recoil {
    template<std::unsigned_integral CdfType, std::unsigned_integral ValueType,
            std::unsigned_integral RansStateType, std::unsigned_integral RansBitstreamType,
            uint8_t ProbBits, RansStateType RenormLowerBound, uint8_t WriteBits, size_t NInterleaved>
    struct SplitCuda {
        using MyRans = Rans<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
        using MySplit = RansSplitsMetadata<CdfType, ValueType, RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits, NInterleaved>::Split;

        size_t cutPosition; // TODO: need to verify this is actually safe!
        cuda::std::array<MyRans, NInterleaved> intermediateRans;
        cuda::std::array<size_t, NInterleaved> startSymbolGroupIds;
        size_t minSymbolGroupId, maxSymbolGroupId;

        explicit SplitCuda(const MySplit &split) : cutPosition(split.cutPosition),
            minSymbolGroupId(split.minSymbolGroupId()), maxSymbolGroupId(split.maxSymbolGroupId()) {
            std::copy(split.intermediateRans.begin(), split.intermediateRans.end(), intermediateRans.begin());
            std::copy(split.startSymbolGroupIds.begin(), split.startSymbolGroupIds.end(), startSymbolGroupIds.begin());
        }
    };
}

#endif //RECOIL_RANS_CODED_DATA_CUDA_H
