#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"
#include "recoil/multithread/rans_split_encoder.h"
#include "recoil/multithread/rans_split_decoder.h"

#include <array>
#include <iostream>

using namespace Recoil;

const int numval = 88 * 2;
const int numsplits = 3;

std::array<CdfType, 5> cdfVal = {0, 1, 65533, 65534, 65535};
std::array<ValueType, numval> values = {
        0, 1, 2, 3, 2, 3, 0, 0,
        1, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 2, 3, 0, 0,
        1, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 2, 3, 0, 0,
        1, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 2, 3, 0, 0,
        1, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 2, 3, 0, 0,
        1, 1, 2, 3, 0, 1, 2, 3,
        1, 2, 3, 3, 2, 3, 3, 3,
};

int main() {
    Cdf cdf(cdfVal);

    for (int i = 0; i < values.size(); i ++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    RansSplitEncoder splitEnc((std::array<Rans32<16>, 4>{}));
    splitEnc.getEncoder().buffer(values, cdf);
    auto result = splitEnc.flushSplits<numsplits>();

    RansSplitDecoder splitDec(result);
    for (int s = 0; s < numsplits; s++) {
        auto decoded = splitDec.decodeSplit(s, cdf);
        for (int i = 0; i < decoded.size(); i ++) {
            std::cout << decoded[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}