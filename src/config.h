#ifndef CONFIG_H
#define CONFIG_H

#include <optional>
#include <random>

namespace hnsw {
    struct HnswConfig {
        size_t capacity = 1024;

        size_t efConstruction = 128;
        size_t M_ = 16;
        size_t M0_ = 2 * M_;

        size_t ef = 64;

        std::optional<std::mt19937::result_type> layerGenSeed = std::nullopt;
        // rate of layer exponential distribution; corresponds to 1/mL for mL as in the paper
        double layerExpLambda = log(static_cast<double>(M_));

        bool extendCandidates = false;
        bool keepPrunedConnections = false;
    };
} // namespace  hnsw
#endif // CONFIG_H