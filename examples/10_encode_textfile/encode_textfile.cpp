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
using ValueType = uint8_t;

int main(int argc, const char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [textfile] [n_splits] [output_prefix]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);

    auto nSplits = std::stoull(argv[2]);
    auto outputPrefix = std::string(argv[3]);

    auto cdfVec = buildCdfFromString(text, ProbBits);

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(cdfVec.size(), 0);
    auto cdfOffset = pool.insertCdf(cdfVec);

    RansSplitEncoder enc((std::array<Rans32<ValueType, ProbBits>, NInterleaved>{}), pool);
    auto symbols = stringToSymbols<ValueType>(text);
    enc.getEncoder().buffer(symbols, cdfOffset);
    auto result = enc.flushSplits(nSplits);

    SplitsMetadataEncoder metadataEnc(result.first, result.second);
    auto bitstream = metadataEnc.combine();

    writeSpanToFile(outputPrefix, std::span{bitstream});
    writeSpanToFile(outputPrefix + ".cdf", std::span{cdfVec});

    return 0;
}