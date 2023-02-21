#ifndef RECOIL_EXAMPLE_PROFILING_H
#define RECOIL_EXAMPLE_PROFILING_H

#include <functional>

namespace Recoil::Examples {
    unsigned int timeIt(const std::function<void()>& func);
}

#endif //RECOIL_EXAMPLE_PROFILING_H
