#include "cdf_utils.h"
#include "file.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/multithread/rans_split_encoder.h"
#include "recoil/cuda/rans_split_decoder_cuda.cuh"

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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [textfile] [nsplit]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);
    auto nSplit = std::stoull(argv[2]);
    std::cout << "File size: " << text.length() << " bytes" << std::endl;

    auto cdfVec = buildCdfFromString(text, ProbBits);
    auto lutVec = LutBuilder<CdfType, ValueType, ProbBits, LutGranularity>::buildLut(std::span{cdfVec});

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(cdfVec.size(), lutVec.size());
    auto cdfOffset = pool.insertCdf(cdfVec);
    auto lutOffset = pool.insertLut(lutVec);

    RansSplitEncoder enc((std::array<Rans32<ValueType, ProbBits>, NInterleaved>{}), pool);
    auto symbols = stringToSymbols<ValueType>(text);
    enc.getEncoder().buffer(symbols, cdfOffset);
    auto result = enc.flushSplits(nSplit);

    RansSplitDecoderCuda splitDecoderCuda(result.first, result.second, pool);
    auto decoded = splitDecoderCuda.decodeAll(cdfOffset, lutOffset);
    //for (auto s:decoded) std::cout<<s;

    auto elapsed = splitDecoderCuda.getLastDuration();
    std::cout << "Time: " << elapsed << "us" << std::endl;
    std::cout << "Throughput: " << text.length() / (elapsed / 1000000.0) / 1024 / 1024 << " MB/s" << std::endl;

    if (std::equal(symbols.begin(), symbols.end(), decoded.begin())) {
        std::cout << "CUDA Decoding success!" << std::endl;
    } else {
        std::cerr << "CUDA Decoding failed!" << std::endl;
    }

    return 0;
}