#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"

#include "recoil/lib/cdf.h"
#include "recoil/rans_encoder.h"
#include "recoil/cuda/rans_decoder_cuda.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>
#include <span>
#include <cuda.h>
#include <cuda/std/array>

using namespace Recoil;
using namespace Recoil::Examples;

const uint8_t ProbBits = 12;
const size_t NInterleaved = 32;

CUDA_GLOBAL void testCudaDecode (auto *bitstreamGpu,
                                      uint32_t offset,
                                      cuda::std::array<Rans32<ProbBits>, NInterleaved> rans,
                                      CdfType *cdfGpu, ValueType *lutGpu, uint32_t symbolCount) {
    RansDecoderCuda decoderCuda(bitstreamGpu, offset, rans[threadIdx.x]);
    decoderCuda.decode(cdfGpu, lutGpu, symbolCount);
}

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

    RansEncoder enc((std::array<Rans32<ProbBits>, NInterleaved>{}));
    std::vector<ValueType> symbols{text.begin(), text.end()};
    enc.buffer(symbols, cdf);
    auto result = enc.flush();

    decltype(result.bitstream.data()) bitstreamGpu;
    auto bitstreamSize = result.bitstream.size() * sizeof(decltype(result.bitstream[0]));
    cudaMalloc(&bitstreamGpu, bitstreamSize);
    cudaMemcpy(bitstreamGpu, result.bitstream.data(), bitstreamSize, cudaMemcpyHostToDevice);

    cuda::std::array<uint32_t, 1> offsets{ static_cast<uint32_t>(result.bitstream.size() - 1) };
    cuda::std::array<Rans32<ProbBits>, NInterleaved> rans;
    for (int i = 0; i < NInterleaved; i++) rans[i] = result.finalRans[i];

    CdfType *cdfGpu;
    ValueType *lutGpu;
    cudaMalloc(&cdfGpu, cdfVec.size() * sizeof(CdfType));
    cudaMalloc(&lutGpu, lutVec.size() * sizeof(ValueType));
    cudaMemcpy(cdfGpu, cdfVec.data(), cdfVec.size() * sizeof(CdfType), cudaMemcpyHostToDevice);
    cudaMemcpy(lutGpu, lutVec.data(), lutVec.size() * sizeof(ValueType), cudaMemcpyHostToDevice);

    testCudaDecode<<<1, 32>>>(bitstreamGpu, offsets[0], rans, cdfGpu, lutGpu, result.symbolCount);
    cudaDeviceSynchronize();
}