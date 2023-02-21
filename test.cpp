#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"
#include "recoil/multithread/rans_split_encoder.h"
#include "recoil/multithread/rans_split_decoder.h"
#include "recoil/simd/rans_decoder_avx2_32x8n.h"

#include <array>
#include <iostream>

using namespace Recoil;

const int numval = 83;

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
        1, 2, 3
};

int main() {
    Cdf cdf(cdfVal);

    for (int i = 0; i < values.size(); i ++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    RansEncoder enc((std::array<Rans32<16>, 8>{}));
    enc.buffer(values, cdf);

    auto result = enc.flush();

    RansDecoder dec((std::span{result.bitstream}), result.finalRans);
    auto decoded = dec.decode(cdf, numval);

    for (int i = 0; i < decoded.size(); i ++) {
        std::cout << decoded[i] << " ";
    }
    std::cout << std::endl;

    RansDecoder_AVX2_32x8n dec2((std::span{result.bitstream}), result.finalRans);
    auto decoded2 = dec2.decode(cdf, numval);

    for (int i = 0; i < decoded2.size(); i ++) {
        std::cout << decoded2[i] << " ";
    }
    std::cout << std::endl;

    RansSplitEncoder splitEnc((std::array<Rans32<16>, 8>{}));
    splitEnc.flushSplits<4>();

    RansCodedDataWithSplits<uint32_t, uint16_t, 16, 1ul<<16, 16, 8, 1> data{
        numval,
        result.bitstream,
        result.finalRans,
        {0, result.finalRans, {0, 0, 0, 0, 0, 0, 0, 0}}
    };

    RansSplitDecoder splitDec(data);
    splitDec.decodeSplit(0, cdf);

    return 0;
}