#include "cdf_utils.h"
#include "file.h"

#include "recoil/lib/cdf.h"
#include "recoil/multithread/rans_split_encoder.h"
#include "recoil/cuda/rans_split_decoder_cuda.cuh"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

using namespace Recoil;
using namespace Recoil::Examples;

const uint8_t ProbBits = 12;
const size_t NInterleaved = 32;
const size_t NSplit = 136;

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [textfile]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);
    std::cout << "File size: " << text.length() << " bytes" << std::endl;

    auto cdfVec = buildCdfFromString(text, ProbBits);
    auto lutVec = Cdf::buildLut<ProbBits>(std::span{cdfVec});
    std::span cdfSpan{cdfVec};
    std::span lutSpan{lutVec};
    Cdf cdf(cdfSpan, lutSpan);

    RansSplitEncoder enc((std::array<Rans32<ProbBits>, NInterleaved>{}));
    std::vector<ValueType> symbols{text.begin(), text.end()};
    enc.getEncoder().buffer(symbols, cdf);
    auto result = enc.flushSplits<NSplit>(/*SplitStrategy::EqualBitstreamLength*/);

    CdfType *cdfGpu;
    ValueType *lutGpu;
    cudaMalloc(&cdfGpu, cdfVec.size() * sizeof(CdfType));
    cudaMalloc(&lutGpu, lutVec.size() * sizeof(ValueType));
    cudaMemcpy(cdfGpu, cdfVec.data(), cdfVec.size() * sizeof(CdfType), cudaMemcpyHostToDevice);
    cudaMemcpy(lutGpu, lutVec.data(), lutVec.size() * sizeof(ValueType), cudaMemcpyHostToDevice);

    RansSplitDecoderCuda splitDecoderCuda(result);
    auto decoded = splitDecoderCuda.decodeAll(cdfGpu, lutGpu);

    /*for (int i = 0; i < decoded.size(); i ++) {
        printf("%c", decoded[i]);
    }*/

    if (std::equal(symbols.begin(), symbols.end(), decoded.begin())) {
        std::cout << "CUDA Decoding success!" << std::endl;
    } else {
        std::cerr << "CUDA Decoding failed!" << std::endl;
    }

    return 0;
}