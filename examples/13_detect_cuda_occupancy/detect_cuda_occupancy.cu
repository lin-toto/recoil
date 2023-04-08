#include "recoil/cuda/rans_split_decoder_cuda.cuh"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

using namespace Recoil;

int main(int argc, const char **argv) {
    const auto maxOccupancySplits = RansSplitDecoderCuda<uint16_t, uint8_t, uint32_t, uint16_t, 12, 1u << 16, 16, 1, 32>::estimateMaxOccupancySplits();
    std::cout << "{\"occupancy\": " << maxOccupancySplits << "}" << std::endl;

    return 0;
}