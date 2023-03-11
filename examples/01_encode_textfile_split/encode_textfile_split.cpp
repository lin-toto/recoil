#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/multithread/rans_split_encoder.h"
#include "recoil/multithread/rans_split_decoder.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>
#include <future>

using namespace Recoil;
using namespace Recoil::Examples;

const uint8_t ProbBits = 12;
const uint8_t LutGranularity = 1;
const size_t NInterleaved = 32;
const size_t NSplit = 32;

using CdfType = uint16_t;
using ValueType = uint8_t;

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [textfile]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);
    std::cout << "File size: " << text.length() << " bytes" << std::endl;

    auto cdfVec = buildCdfFromString(text, ProbBits);
    auto lutVec = LutBuilder<CdfType, ValueType, ProbBits, LutGranularity>::buildLut(std::span{cdfVec});

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(cdfVec.size(), lutVec.size());
    auto cdfOffset = pool.insertCdf(cdfVec);
    auto lutOffset = pool.insertLut(lutVec);

    RansSplitEncoder enc((std::array<Rans32<ValueType, ProbBits>, NInterleaved>{}), pool);
    auto symbols = stringToSymbols<ValueType>(text);
    enc.getEncoder().buffer(symbols, cdfOffset);
    auto result = enc.flushSplits<NSplit>();

    std::array<std::vector<ValueType>, NSplit> decoded;

    RansSplitDecoder dec(result, pool);

    std::array<std::future<unsigned int>, NSplit> tasks;
    for (int i = 0; i < NSplit; i++) {
        tasks[i] = std::async(std::launch::async, [i, &dec, &decoded, cdfOffset, lutOffset] {
            auto time = timeIt([&]() { decoded[i] = dec.decodeSplit(i, cdfOffset, lutOffset); });
            return time;
        });
    }

    std::vector<ValueType> allDecoded;
    double sumThroughput = 0;
    for (int i = 0; i < NSplit; i++) {
        tasks[i].wait();
        auto time = tasks[i].get();
        auto t = decoded[i].size() / (time / 1000000.0) / 1024 / 1024;
        std::cout << "Multithread Throughput: " << t << " MB/s" << std::endl;
        sumThroughput += t;

        std::copy(decoded[i].begin(), decoded[i].end(), std::back_inserter(allDecoded));
    }
    std::cout << "Sum: " << sumThroughput << " MB/s" << std::endl;

    if (std::equal(symbols.begin(), symbols.end(), allDecoded.begin())) {
        std::cout << "Decoding success!" << std::endl;
    } else {
        std::cerr << "Decoding failed!" << std::endl;
    }

    return 0;
}