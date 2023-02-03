#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"

#include <array>
#include <iostream>

using namespace Recoil;

const int numval = 5;

std::array<CdfType, 5> cdfVal = {0, 1, 65533, 65534, 65535};
std::array<ValueType, numval> values = {0, 1, 2, 2, 3};

int main() {
    Cdf cdf(cdfVal);

    RansEncoder enc((std::array{Rans32<16>()}));
    enc.buffer(values, cdf);

    auto result = enc.flush();

    RansDecoder dec(std::span{result.bitstream}, result.finalRans);
    auto decoded = dec.decode(cdf, numval);

    for (int i = 0; i < numval; i ++) {
        std::cout << decoded[i] << std::endl;
    }

    return 0;
}