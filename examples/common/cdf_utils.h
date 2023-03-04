#ifndef RECOIL_EXAMPLE_CDF_UTILS_H
#define RECOIL_EXAMPLE_CDF_UTILS_H

#include "recoil/rans.h"
#include <vector>
#include <string_view>

namespace Recoil::Examples {
    std::vector<CdfType> buildCdfFromString(std::string_view str, uint8_t probBits);
    void fixCdfZerosByStealing(std::vector<CdfType> &cdf);
};

#endif //RECOIL_EXAMPLE_CDF_UTILS_H
