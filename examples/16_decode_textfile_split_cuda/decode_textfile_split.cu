#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/split/rans_split_encoder.h"
#include "recoil/cuda/rans_split_decoder_cuda.cuh"
#include "recoil/split/bitstream_generation/splits_metadata_encoder.h"
#include "recoil/split/bitstream_generation/splits_metadata_decoder.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

using namespace Recoil;
using namespace Recoil::Examples;

const uint8_t ProbBits = 12;
const uint8_t LutGranularity = 1;
const size_t NInterleaved = 32;

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

    RansSplitDecoderCuda splitDecoderCuda(result.first, result.second, pool);
    auto decoded = splitDecoderCuda.decodeAll(cdfOffset, lutOffset);

    auto text = stringToSymbols<uint8_t>(readFile(argv[3]));
    bool correct = std::equal(decoded.begin(), decoded.end(), text.begin());

    auto elapsed = splitDecoderCuda.getLastDuration();
    std::cout << jsonOutput(correct, nSplit, decoded.size(), bitstream.size() * sizeof(uint16_t), elapsed);

    return 0;
}