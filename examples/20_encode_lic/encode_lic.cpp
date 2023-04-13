#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"
#include "params.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/split/rans_split_encoder.h"
#include "recoil/split/bitstream_generation/splits_metadata_encoder.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

using namespace Recoil;
using namespace Recoil::Examples;

using CdfType = uint16_t;
using ValueType = uint16_t;

int main(int argc, const char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " [img_data.txt] [cdf_indices.txt] [cdf.txt] [n_splits] [output_bin]" << std::endl;
        return 1;
    }

    auto imgSymbols = readVectorFromTextFile<ValueType>(argv[1]);
    auto rawCdfIndices = readVectorFromTextFile<ValueType>(argv[2]);

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(40000, 0); // Just give it enough memory

    std::vector<CdfLutOffsetType> cdfIndicesMap;
    std::ifstream cdfTxt(argv[3]);
    if (!cdfTxt.good()) [[unlikely]] throw std::runtime_error("Error reading cdf text file");
    while (!cdfTxt.eof()) {
        int count;
        cdfTxt >> count;

        auto cdfVec = readVectorFromTextStream<CdfType>(cdfTxt, count);
        auto cdfIndex = pool.insertCdf(cdfVec);
        cdfIndicesMap.push_back(cdfIndex);
    }
    std::vector<CdfLutOffsetType> cdfIndices(rawCdfIndices.size());
    std::transform(rawCdfIndices.begin(), rawCdfIndices.end(), cdfIndices.begin(), [&cdfIndicesMap](auto v) {
        return cdfIndicesMap[v];
    });

    auto nSplits = std::stoull(argv[4]);
    auto outputPrefix = std::string(argv[5]);

    RansSplitEncoder enc((std::array<Rans32<ValueType, ProbBits>, NInterleaved>{}), pool);
    enc.getEncoder().buffer(imgSymbols, cdfIndices);
    auto result = enc.flushSplits(nSplits);

    SplitsMetadataEncoder metadataEnc(result.first, result.second);
    auto bitstream = metadataEnc.combine();

    writeSpanToFile(outputPrefix, std::span{bitstream});

    return 0;
}