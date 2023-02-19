#include "cdf_utils.h"
#include "file_utils.h"

#include "recoil/lib/cdf.h"
#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"

#include <iostream>
#include <vector>
#include <cstdint>

using namespace Recoil;
using namespace Recoil::Examples;

const uint8_t ProbBits = 16;

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [textfile]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);
    auto cdfVec = buildCdfFromString(text, ProbBits);

    Cdf cdf{std::span{cdfVec}};

    RansEncoder enc((std::array{
            Rans32<16>(), Rans32<16>(), Rans32<16>(), Rans32<16>(),
            Rans32<16>(), Rans32<16>(), Rans32<16>(), Rans32<16>()
    }));
    std::vector<ValueType> symbols{text.begin(), text.end()};
    enc.buffer(symbols, cdf);
    auto result = enc.flush();

    RansDecoder dec((std::span{result.bitstream}), result.finalRans);
    auto decoded = dec.decode(cdf, symbols.size());
    if (std::equal(symbols.begin(), symbols.end(), decoded.begin())) {
        std::cout << "Decoding success!" << std::endl;
    } else {
        std::cerr << "Decoding failed!" << std::endl;
    }

    RansDecoder_AVX2_32x8n decAVX2((std::span{result.bitstream}), result.finalRans);
    auto decodedAVX2 = decAVX2.decode(cdf, symbols.size());
    if (std::equal(symbols.begin(), symbols.end(), decodedAVX2.begin())) {
        std::cout << "AVX2 Decoding success!" << std::endl;
    } else {
        std::cerr << "AVX2 Decoding failed!" << std::endl;
    }

    return 0;
}