#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"
#include "latch_backport.h"
#include "params.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/split/rans_split_decoder.h"
#include "recoil/split/bitstream_generation/splits_metadata_decoder.h"

#include <iostream>
#include <cstdint>
#include <array>
#include <future>
#include <chrono>

using namespace Recoil;
using namespace Recoil::Examples;
using namespace std::chrono_literals;

using CdfType = uint16_t;
using ValueType = uint8_t;

int main(int argc, const char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [encoded_textfile] [cdf] [original_file]" << std::endl;
        return 1;
    }

    auto bitstream = readVectorFromFile<uint16_t>(argv[1]);

    auto cdfVec = readVectorFromFile<CdfType>(argv[2]);
    auto lutVec = LutBuilder<CdfType, ValueType, ProbBits, LutGranularity>::buildLut(std::span{cdfVec});

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(cdfVec.size(), lutVec.size());
    auto cdfOffset = pool.insertCdf(cdfVec);
    auto lutOffset = pool.insertLut(lutVec);

    SplitsMetadataDecoder_Rans32<ValueType, ProbBits, NInterleaved> metadataDec(bitstream);
    auto result = metadataDec.decode();
    const auto nSplit = result.second.splits.size();

    RansSplitDecoder dec(result.first, result.second, pool);

    Latch latch, completeLatch(nSplit);
    std::vector<std::future<void>> tasks;
    for (int i = 0; i < nSplit; i++) {
        tasks.push_back(std::async(std::launch::async, [i, &dec, &latch, &completeLatch, cdfOffset, lutOffset] {
            latch.wait();
            dec.decodeSplit(i, cdfOffset, lutOffset);
            completeLatch.count_down();
        }));
    }

    std::this_thread::sleep_for(100ms);

    auto timer = std::chrono::high_resolution_clock::now();
    latch.count_down();
    completeLatch.wait();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timer).count();

    auto text = stringToSymbols<uint8_t>(readFile(argv[3]));
    bool correct = std::equal(dec.result.begin(), dec.result.end(), text.begin());

    std::cout << jsonOutput(correct, nSplit, dec.result.size(), bitstream.size() * sizeof(uint16_t), elapsed);

    return 0;
}