#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"

#include <array>
#include <iostream>

using namespace Recoil;

const int numval = 10;

std::array<CdfType, 5> cdfVal = {0, 1, 65533, 65534, 65535};
std::array<ValueType, numval> values = {0, 1, 2, 3, 0, 1, 2, 3, 1, 1};

int main() {
    Cdf cdf(cdfVal);

    RansEncoder enc((std::array{
        Rans32<16>(), Rans32<16>(), Rans32<16>(), Rans32<16>(),
        Rans32<16>(), Rans32<16>(), Rans32<16>(), Rans32<16>()
    }));
    enc.buffer(values, cdf);

    auto result = enc.flush();

    RansDecoder dec(std::span{result.bitstream}, result.finalRans);
    auto decoded = dec.decode(cdf, numval);

    for (int i = 0; i < decoded.size(); i ++) {
        std::cout << decoded[i] << std::endl;
    }

    RansDecoder_AVX2_32x8n dec2(std::span{result.bitstream}, result.finalRans);
    auto decoded2 = dec2.decode(cdf, numval);

    for (int i = 0; i < decoded2.size(); i ++) {
        std::cout << decoded2[i] << std::endl;
    }

    return 0;
}