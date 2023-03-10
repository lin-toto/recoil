#ifndef RECOIL_LOOKUP_MODE_H
#define RECOIL_LOOKUP_MODE_H

#include <cstdint>

namespace Recoil {
    template<uint8_t LutGranularity>
    concept CdfOnlyGranularity = LutGranularity == 0;

    template<uint8_t LutGranularity>
    concept LutOnlyGranularity = LutGranularity == 1;

    template<uint8_t LutGranularity>
    concept MixedGranularity = LutGranularity > 1;
}

#endif //RECOIL_LOOKUP_MODE_H
