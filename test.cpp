#include "recoil/rans_encoder.h"
#include "recoil/rans_decoder.h"

#include <array>

using namespace Recoil;

int main() {
    RansEncoder enc((std::array{Rans64<16>()}));
}