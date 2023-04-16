#ifndef RECOIL_EXAMPLES_PARAMS_H
#define RECOIL_EXAMPLES_PARAMS_H

#include <cstdint>

#ifdef PROB_BITS
const uint8_t ProbBits = PROB_BITS;
#else
const uint8_t ProbBits = 16;
#endif

#ifdef LUT_GRANULARITY
const uint8_t LutGranularity = LUT_GRANULARITY;
#else
const uint8_t LutGranularity = 1;
#endif

const std::size_t NInterleaved = 32;

#endif //RECOIL_EXAMPLES_PARAMS_H
