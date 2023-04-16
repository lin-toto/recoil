#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"
#include "params.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/cuda/rans_symbol_split_decoder_cuda.cuh"
#include "recoil/split/bitstream_generation/symbol_splits_bitstream_decoder.h"
#include "recoil/rans_decoder.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

using namespace Recoil;
using namespace Recoil::Examples;

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

    RansSymbolSplitDecoderCuda splitDecoderCuda(result, pool);
    auto decoded = splitDecoderCuda.decodeAll(cdfIndices, lutIndices);

    auto imgSymbols = readVectorFromTextFile<ValueType>(argv[4]);
    bool correct = std::equal(decoded.begin(), decoded.end(), imgSymbols.begin());

    auto elapsed = splitDecoderCuda.getLastDuration();
    std::cout << jsonOutput(correct, nSplit, decoded.size() * 2, bitstream.size() * sizeof(uint16_t), elapsed);

    return 0;
}