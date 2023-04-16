#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"
#include "latch_backport.h"
#include "params.h"

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

using CdfType = uint16_t;
using ValueType = uint16_t;

int main(int argc, const char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " [encoded_lic.bin] [cdf_indices.txt] [cdf.txt] [original_data.txt]" << std::endl;
        return 1;
    }

    auto bitstream = readVectorFromFile<uint16_t>(argv[1]);
    auto rawCdfIndices = readVectorFromTextFile<CdfLutOffsetType>(argv[2]);

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(40000, (1 << (ProbBits - LutGranularity + 1)) * 65); // Just give it enough memory

    std::vector<CdfLutOffsetType> cdfIndicesMap, lutIndicesMap;
    std::ifstream cdfTxt(argv[3]);
    if (!cdfTxt.good()) [[unlikely]] throw std::runtime_error("Error reading cdf text file");
    while (!cdfTxt.eof()) {
        int count;
        cdfTxt >> count;

        auto cdfVec = readVectorFromTextStream<CdfType>(cdfTxt, count);
        auto lutVec = LutBuilder<CdfType, ValueType, ProbBits, LutGranularity>::buildLut(std::span{cdfVec});

        auto cdfIndex = pool.insertCdf(cdfVec);
        auto lutIndex = pool.insertLut(lutVec);

        cdfIndicesMap.push_back(cdfIndex);
        lutIndicesMap.push_back(lutIndex);
    }
    pool.insertCdf({0});
    std::vector<CdfLutOffsetType> cdfIndices(rawCdfIndices.size()), lutIndices(rawCdfIndices.size());
    std::transform(rawCdfIndices.begin(), rawCdfIndices.end(), cdfIndices.begin(), [&cdfIndicesMap](auto v) {
        return cdfIndicesMap[v];
    });
    std::transform(rawCdfIndices.begin(), rawCdfIndices.end(), lutIndices.begin(), [&lutIndicesMap](auto v) {
        return lutIndicesMap[v];
    });

    SymbolSplitsBitstreamDecoder_Rans32<ValueType, ProbBits, NInterleaved> metadataDec(bitstream);
    auto result = metadataDec.decode();
    const auto nSplit = result.size();

    RansSymbolSplitDecoder dec(result, pool);

    Latch latch, completeLatch(nSplit);
    std::vector<std::future<void>> tasks;
    for (int i = 0; i < nSplit; i++) {
        tasks.push_back(std::async(std::launch::async,
                                   [i, nSplit, &dec, &latch, &completeLatch, &cdfIndices, &lutIndices, &result] {
            auto symbolCount = result[i].symbolCount;
            auto *myCdfIndices = new (std::align_val_t(32)) CdfLutOffsetType[symbolCount];
            auto *myLutIndices = new (std::align_val_t(32)) CdfLutOffsetType[symbolCount];

            auto offset = result[0].symbolCount * i;
            std::copy(cdfIndices.begin() + offset, cdfIndices.begin() + offset + symbolCount, myCdfIndices);
            std::copy(lutIndices.begin() + offset, lutIndices.begin() + offset + symbolCount, myLutIndices);

            latch.wait();
            dec.decodeSplit(i,
                            std::span{myCdfIndices, symbolCount},
                            std::span{myLutIndices, symbolCount});
            completeLatch.count_down();

            delete[] myCdfIndices;
            delete[] myLutIndices;
        }));
    }

    std::this_thread::sleep_for(100ms);

    auto timer = std::chrono::high_resolution_clock::now();
    latch.count_down();
    completeLatch.wait();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timer).count();

    auto imgSymbols = readVectorFromTextFile<ValueType>(argv[4]);
    bool correct = std::equal(dec.result.begin(), dec.result.end(), imgSymbols.begin());

    std::cout << jsonOutput(correct, nSplit, dec.result.size() * 2, bitstream.size() * sizeof(uint16_t), elapsed);

    return 0;
}