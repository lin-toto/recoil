#ifndef RECOIL_EXAMPLE_CDF_UTILS_H
#define RECOIL_EXAMPLE_CDF_UTILS_H

#include "recoil/rans.h"
#include <vector>
#include <string_view>

namespace Recoil::Examples {
    // TODO: eventually have to move these to the main header library!
    std::vector<CdfType> buildCdfFromString(std::string_view str, uint8_t probBits);
    void fixCdfZerosByStealing(std::vector<CdfType> &cdf);
    std::vector<ValueType> buildLut(std::span<CdfType> cdf, uint8_t probBits, uint8_t lutGranularity);
};

#endif //RECOIL_EXAMPLE_CDF_UTILS_H
