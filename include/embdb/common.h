#ifndef COMMON_H
#define COMMON_H

#include <chrono>
#include <vector>

namespace hnsw {
    using IdType = size_t;
    using IdVector = std::vector<IdType>;

    const IdType INVALID_ID = static_cast<IdType>(-1);

    template<class Clock = std::chrono::high_resolution_clock>
    class StopWatch {
     public:
        StopWatch() {
            reset();
        }

        template<class Unit = std::chrono::microseconds>
        long elapsed() const {
            return std::chrono::duration_cast<Unit>(Clock::now() - start).count();
        }

        void reset() {
            start = Clock::now();
        }

     private:
        typename Clock::time_point start;
    };
} // namespace hnsw
#endif // COMMON_H