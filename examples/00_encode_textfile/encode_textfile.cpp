#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"

#include "recoil/lib/cdf.h"
#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

using namespace Recoil;
using namespace Recoil::Examples;

const uint8_t ProbBits = 12;
const size_t NInterleaved = 16;

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [textfile]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);
    std::cout << "File size: " << text.length() << " bytes" << std::endl;

    auto cdfVec = buildCdfFromString(text, ProbBits);
    auto lutVec = Cdf::buildLut<ProbBits>(std::span{cdfVec});
    Cdf cdf((std::span{cdfVec}), (std::span{lutVec}));

    RansEncoder enc((std::array<Rans32<ProbBits>, NInterleaved>{}));
    auto symbols = stringToSymbols(text);
    enc.buffer(symbols, cdf);
    auto result = enc.flush();

    RansDecoder dec((std::span{result.bitstream}), result.finalRans);
    std::vector<ValueType> decoded;
    auto time = timeIt([&]() { decoded = dec.decode(cdf, symbols.size()); });
    if (std::equal(symbols.begin(), symbols.end(), decoded.begin())) {
        std::cout << "Decoding success! Time: " << time << "us" << std::endl;
    } else {
        std::cerr << "Decoding failed!" << std::endl;
    }
    std::cout << "Throughput: " << text.length() / (time / 1000000.0) / 1024 / 1024 << " MB/s" << std::endl;

    RansDecoder_AVX2_32x8n decAVX2((std::span{result.bitstream}), result.finalRans);
    time = timeIt([&]() { decoded = decAVX2.decode(cdf, symbols.size()); });
    if (std::equal(symbols.begin(), symbols.end(), decoded.begin())) {
        std::cout << "AVX2 Decoding success! Time: " << time << "us" << std::endl;
    } else {
        std::cerr << "AVX2 Decoding failed!" << std::endl;
    }
    std::cout << "Throughput: " << text.length() / (time / 1000000.0) / 1024 / 1024 << " MB/s" << std::endl;

    return 0;
}