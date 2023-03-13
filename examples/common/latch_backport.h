#ifndef RECOIL_LATCH_BACKPORT_H
#define RECOIL_LATCH_BACKPORT_H

#include <condition_variable>
#include <atomic>
#include <mutex>

/*
 * Backport of std::latch since many compilers still don't support it.
 * Implementation from https://codereview.stackexchange.com/questions/269344/c11-revised-stdlatch-implementation
 */
namespace Recoil::Examples {
    class Latch {
        std::atomic<std::ptrdiff_t> counter_;
        mutable std::condition_variable cv_;
        mutable std::mutex mut_;

    public:
        explicit Latch(std::ptrdiff_t const def = 1) : counter_(def) {}

        inline void count_down(std::ptrdiff_t const n = 1) {
            counter_ -= n;
            cv_.notify_all();
        }

        inline void wait() const {
            if (counter_.load(std::memory_order_relaxed) == 0) return;
            std::unique_lock<std::mutex> lock(mut_);
            cv_.wait(lock, [this] { return counter_.load(std::memory_order_relaxed) == 0; });
        }

        inline bool try_wait() const noexcept {
            return counter_.load(std::memory_order_relaxed) == 0;
        }

        inline void arrive_and_wait(std::ptrdiff_t const n = 1) {
            count_down(n);
            wait();
        }

        inline static constexpr std::ptrdiff_t max() noexcept {
            return std::numeric_limits<std::ptrdiff_t>::max();
        }
    };
}

#endif //RECOIL_LATCH_BACKPORT_H
