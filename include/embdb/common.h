#ifndef COMMON_H
#define COMMON_H

#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/fmt/ostr.h"

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

    bool initLogging() {
        auto console = spdlog::stdout_color_mt("console");
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S,%e] [tid: %t] [%^%l%$] %s.%# : %v");
        spdlog::set_default_logger(console);
        spdlog::set_level(spdlog::level::debug);
        return true;
    }
} // namespace hnsw
#endif // COMMON_H