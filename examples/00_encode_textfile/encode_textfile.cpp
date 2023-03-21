#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"
#include "recoil/simd/rans_decoder_avx2_32x32.h"
//#include "recoil/simd/rans_decoder_avx512_32x16n.h"
//#include "recoil/simd/rans_decoder_avx512_32x32.h"

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

    RansEncoder enc((std::array<Rans32<ValueType, ProbBits>, NInterleaved>{}), pool);
    auto symbols = stringToSymbols<ValueType>(text);
    enc.buffer(symbols, cdfOffset);
    auto result = enc.flush();

    RansDecoder dec((result.getRealBitstream()), result.finalRans, pool);
    std::vector<ValueType> decoded;
    auto time = timeIt([&]() { decoded = dec.decode(cdfOffset, lutOffset, symbols.size()); });
    if (std::equal(symbols.begin(), symbols.end(), decoded.begin())) {
        std::cout << "Decoding success! Time: " << time << "us" << std::endl;
    } else {
        std::cerr << "Decoding failed!" << std::endl;
    }
    std::cout << "Throughput: " << text.length() / (time / 1000000.0) / 1024 / 1024 << " MB/s" << std::endl;

    RansDecoder_AVX2_32x32 decAVX2(result.getRealBitstream(), result.finalRans, pool);
    time = timeIt([&]() { decoded = decAVX2.decode(cdfOffset, lutOffset, symbols.size()); });
    if (std::equal(symbols.begin(), symbols.end(), decoded.begin())) {
        std::cout << "AVX2 Decoding success! Time: " << time << "us" << std::endl;
    } else {
        std::cerr << "AVX2 Decoding failed!" << std::endl;
    }
    std::cout << "Throughput: " << text.length() / (time / 1000000.0) / 1024 / 1024 << " MB/s" << std::endl;

    return 0;
}