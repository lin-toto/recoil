#ifndef RECOIL_EXAMPLE_PROFILING_H
#define RECOIL_EXAMPLE_PROFILING_H

#include <functional>
#include <string>

namespace Recoil::Examples {
    unsigned int timeIt(const std::function<void()>& func);
    std::string jsonOutput(size_t originalSize, size_t compressedSize, unsigned int elapsed);
}

#endif //RECOIL_EXAMPLE_PROFILING_H
