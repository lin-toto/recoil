#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"
#include "latch_backport.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/split/rans_split_encoder.h"
#include "recoil/split/rans_split_decoder.h"
#include "recoil/split/bitstream_generation/splits_metadata_encoder.h"
#include "recoil/split/bitstream_generation/splits_metadata_decoder.h"

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
        std::cerr << "Usage: " << argv[0] << " [textfile] [nsplit]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);
    auto nSplit = std::stoull(argv[2]);
    std::cout << "File size: " << text.length() << " bytes" << std::endl;

    auto cdfVec = buildCdfFromString(text, ProbBits);
    auto lutVec = LutBuilder<CdfType, ValueType, ProbBits, LutGranularity>::buildLut(std::span{cdfVec});

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(cdfVec.size(), lutVec.size());
    auto cdfOffset = pool.insertCdf(cdfVec);
    auto lutOffset = pool.insertLut(lutVec);

    RansSplitEncoder enc((std::array<Rans32<ValueType, ProbBits>, NInterleaved>{}), pool);
    auto symbols = stringToSymbols<ValueType>(text);
    enc.getEncoder().buffer(symbols, cdfOffset);
    auto result = enc.flushSplits(nSplit);

    SplitsMetadataEncoder metadataEnc(result.first, result.second);
    auto bitstream = metadataEnc.combine();

    SplitsMetadataDecoder_Rans32<ValueType, ProbBits, NInterleaved> metadataDec(bitstream);
    auto result2 = metadataDec.decode();

    RansSplitDecoder dec(result2.first, result2.second, pool);

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
    std::cout << "Time: " << elapsed << "us" << std::endl;
    std::cout << "Throughput: " << text.length() / (elapsed / 1000000.0) / 1024 / 1024 << " MB/s" << std::endl;

    if (std::equal(symbols.begin(), symbols.end(), dec.result.begin())) {
        std::cout << "Decoding success!" << std::endl;
    } else {
        std::cerr << "Decoding failed!" << std::endl;
    }

    return 0;
}