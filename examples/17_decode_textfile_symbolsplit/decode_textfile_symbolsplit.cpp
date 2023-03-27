#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"
#include "latch_backport.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/split/rans_symbol_split_decoder.h"
#include "recoil/split/bitstream_generation/symbol_splits_bitstream_decoder.h"

#include <iostream>
#include <cstdint>
#include <array>
#include <future>
#include <chrono>

using namespace Recoil;
using namespace Recoil::Examples;
using namespace std::chrono_literals;

const uint8_t ProbBits = 12;
const uint8_t LutGranularity = 1;
const size_t NInterleaved = 32;

using CdfType = uint16_t;
using ValueType = uint8_t;

int main(int argc, const char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [encoded_textfile] [cdf]" << std::endl;
        return 1;
    }

    auto bitstream = readVectorFromFile<uint16_t>(argv[1]);

    auto cdfVec = readVectorFromFile<CdfType>(argv[2]);
    auto lutVec = LutBuilder<CdfType, ValueType, ProbBits, LutGranularity>::buildLut(std::span{cdfVec});

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(cdfVec.size(), lutVec.size());
    auto cdfOffset = pool.insertCdf(cdfVec);
    auto lutOffset = pool.insertLut(lutVec);

    SymbolSplitsBitstreamDecoder_Rans32<ValueType, ProbBits, NInterleaved> metadataDec(bitstream);
    auto result = metadataDec.decode();
    auto nSplit = result.size();

    RansSymbolSplitDecoder dec(result, pool);

    Latch latch;
    std::vector<std::future<void>> tasks;
    for (int i = 0; i < nSplit; i++) {
        tasks.push_back(std::async(std::launch::async, [i, &dec, &latch, cdfOffset, lutOffset] {
            latch.wait();
            dec.decodeSplit(i, cdfOffset, lutOffset);
        }));
    }

    std::this_thread::sleep_for(100ms);

    auto timer = std::chrono::high_resolution_clock::now();
    latch.count_down();
    for (int i = 0; i < nSplit; i++) tasks[i].wait();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timer).count();

    std::cout << jsonOutput(nSplit, dec.result.size(), bitstream.size() * sizeof(uint16_t), elapsed);

    return 0;
}