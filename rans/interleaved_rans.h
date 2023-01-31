#ifndef RECOIL_INTERLEAVED_RANS_H
#define RECOIL_INTERLEAVED_RANS_H

#include <array>
#include "rans/rans.h"

template <typename MyRans, int nInterleaved>
class InterleavedRans {

protected:
    std::array<MyRans, nInterleaved> rans;
};

#endif //RECOIL_INTERLEAVED_RANS_H
