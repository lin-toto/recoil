#include "rans/rans_encoder.h"
#include "rans/rans_decoder.h"

#include <array>

using namespace Recoil;

int main() {
    RansEncoder enc((std::array{Rans64<16>()}));
}