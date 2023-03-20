#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"
#include "recoil/split/rans_split_encoder.h"
#include "recoil/split/rans_split_decoder.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/symbol_lookup/symbol_lookup.h"

#include <array>
#include <iostream>

using namespace Recoil;

const int numval = 88;
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

    RansSplitEncoder splitEnc((std::array<Rans32<16>, 8>{}));
    splitEnc.getEncoder().buffer(values, cdf);
    auto result = splitEnc.flushSplits<numsplits>(Recoil::EqualBitstreamLength);

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