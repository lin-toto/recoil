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
    auto cdfLutSize = rawCdfIndices.size();
    auto *cdfIndices = new (std::align_val_t(32)) CdfLutOffsetType[cdfLutSize];
    auto *lutIndices = new (std::align_val_t(32)) CdfLutOffsetType[cdfLutSize];
    std::transform(rawCdfIndices.begin(), rawCdfIndices.end(), cdfIndices, [&cdfIndicesMap](auto v) {
        return cdfIndicesMap[v];
    });
    std::transform(rawCdfIndices.begin(), rawCdfIndices.end(), lutIndices, [&lutIndicesMap](auto v) {
        return lutIndicesMap[v];
    });

    SplitsMetadataDecoder_Rans32<ValueType, ProbBits, NInterleaved> metadataDec(bitstream);
    auto result = metadataDec.decode();
    const auto nSplit = result.second.splits.size();

    RansSplitDecoder dec(result.first, result.second, pool);

    Latch latch, completeLatch(nSplit);
    std::vector<std::future<void>> tasks;
    for (int i = 0; i < nSplit; i++) {
        tasks.push_back(std::async(std::launch::async, [i, &dec, &latch, &completeLatch, &cdfIndices, &lutIndices, cdfLutSize] {
            latch.wait();
            dec.decodeSplit(i, std::span{cdfIndices, cdfLutSize}, std::span{lutIndices, cdfLutSize});
            completeLatch.count_down();
        }));
    }

    //dec.decodeSplit(0, std::span{cdfIndices, cdfLutSize}, std::span{lutIndices, cdfLutSize});

    std::this_thread::sleep_for(100ms);

    auto timer = std::chrono::high_resolution_clock::now();
    latch.count_down();
    completeLatch.wait();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timer).count();

    auto imgSymbols = readVectorFromTextFile<ValueType>(argv[4]);
    bool correct = std::equal(dec.result.begin(), dec.result.end(), imgSymbols.begin());
    for (int i = 0; i < 5000; i++) if (imgSymbols[i] != dec.result[i])
        std::cout << i << " " << imgSymbols[i] << " " << dec.result[i] << std::endl;

    std::cout << jsonOutput(correct, nSplit, dec.result.size() * 2, bitstream.size() * sizeof(uint16_t), elapsed);

    return 0;
}