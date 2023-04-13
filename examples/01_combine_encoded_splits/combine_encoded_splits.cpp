#include "file.h"
#include "params.h"

#include "recoil/split/bitstream_generation/splits_metadata_encoder.h"
#include "recoil/split/bitstream_generation/splits_metadata_decoder.h"

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
        std::cerr << "Usage: " << argv[0] << " [input_bitstream] [n_splits] [output_bitstream]" << std::endl;
        return 1;
    }

    auto bitstream = readVectorFromFile<uint16_t>(argv[1]);
    auto newNSplits = std::stoull(argv[2]);

    SplitsMetadataDecoder_Rans32<ValueType, ProbBits, NInterleaved> metadataDec(bitstream);
    auto result = metadataDec.decode();

    result.second.reduceSplitCount(newNSplits);

    SplitsMetadataEncoder metadataEnc(result.first, result.second);
    auto newBitstream = metadataEnc.combine();

    writeSpanToFile(argv[3], std::span{newBitstream});

    return 0;
}