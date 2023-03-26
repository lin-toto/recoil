#include "cdf_utils.h"
#include "file.h"
#include "profiling.h"

#include "recoil/symbol_lookup/cdf_lut_pool.h"
#include "recoil/split/rans_split_encoder.h"
#include "recoil/split/bitstream_generation/splits_metadata_encoder.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

using namespace Recoil;
using namespace Recoil::Examples;

const uint8_t ProbBits = 16;
const uint8_t LutGranularity = 1;
const size_t NInterleaved = 32;

using CdfType = uint16_t;
using ValueType = uint8_t;

int main(int argc, const char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [textfile] [n_splits] [output_prefix]" << std::endl;
        return 1;
    }

    auto text = readFile(argv[1]);
    std::cout << "File size: " << text.length() << " bytes" << std::endl;

    auto nSplits = std::stoull(argv[2]);
    auto outputPrefix = std::string(argv[3]);

    auto cdfVec = buildCdfFromString(text, ProbBits);

    CdfLutPool<CdfType, ValueType, ProbBits, LutGranularity> pool(cdfVec.size(), 0);
    auto cdfOffset = pool.insertCdf(cdfVec);

    auto symbols = stringToSymbols<ValueType>(text);

    auto symbolsPerSplit = saveDiv<size_t>(symbols.size(), nSplits);
    std::vector<uint16_t> bitstream;

    BitsWriter<uint16_t> writer;
    std::vector<size_t> bitstreamLengths;
    for (int splitId = 0; splitId < nSplits; splitId++) {
        RansEncoder enc((std::array<Rans32<ValueType, ProbBits>, NInterleaved>{}), pool);
        enc.buffer(std::span{symbols}.subspan(symbolsPerSplit * splitId, splitId == nSplits - 1 ? std::dynamic_extent : symbolsPerSplit), cdfOffset);
        auto r = enc.flush();

        for (auto rans : r.finalRans) writer.writeData(rans.state, 32);
        std::copy(r.getRealBitstream().begin(), r.getRealBitstream().end(), std::back_inserter(bitstream));
        bitstreamLengths.push_back(r.getRealBitstream().size());
    }

    std::vector<size_t> bitstreamLengthDiffs(bitstreamLengths.size());
    std::transform(bitstreamLengths.begin(), bitstreamLengths.end(), bitstreamLengthDiffs.begin(), [&bitstream, nSplits] (auto l) {
        return static_cast<int32_t>(l) - saveDiv<size_t>(bitstream.size(), nSplits);
    });
    auto len = writer.getActualLength(*std::max_element(
            bitstreamLengthDiffs.begin(), bitstreamLengthDiffs.end(),
            [](const auto& a, const auto& b) { return abs(a) < abs(b); }));;
    writer.template writeLength<int32_t>(len);
    for (auto bitstreamLengthDiff : bitstreamLengthDiffs)
        writer.template writeData<int32_t>(bitstreamLengthDiff, len);
    writer.template write<uint32_t>(symbols.size());
    bitstream.insert(bitstream.begin(), writer.buf.begin(), writer.buf.end());

    writeSpanToFile(outputPrefix + ".bin", std::span{bitstream});
    writeSpanToFile(outputPrefix + ".cdf", std::span{cdfVec});

    return 0;
}