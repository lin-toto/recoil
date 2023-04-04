#include "profiling.h"

#include <chrono>

namespace Recoil::Examples {
    unsigned int timeIt(const std::function<void()>& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count();
    }

    std::string jsonOutput(bool correct, unsigned int nSplit, size_t originalSize, size_t compressedSize, unsigned int elapsed) {
        const char *jsonLiteral = R"({
    "result_correct": %s,
    "n_splits": %u,
    "original_size_bytes": %llu,
    "compressed_size_bytes": %llu,
    "time": %u,
    "throughput_mb": %.2f
})";

        float throughput = originalSize / (elapsed / 1000000.0) / 1024 / 1024;

        char buf[200]; // Probably more than enough?
        snprintf(buf, 200, jsonLiteral, correct ? "true" : "false", nSplit, originalSize, compressedSize, elapsed, throughput);

        return buf;
    }
}
