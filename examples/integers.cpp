#include <iostream>

#include "embdb/hnsw.h"

class IntegerSpace : public hnsw::MetricSpace<int, int> {
 public:
    int distance(const int & obj1, const int & obj2) const override {
        return abs(obj1 - obj2);
    }
};

int main() {
    const size_t k = 4;

    IntegerSpace space;

    hnsw::HnswConfig config;
    config.capacity = 1000000;
    config.efConstruction = 64;
    config.ef = 64;

    using IndexType = hnsw::HnswIndex<int, int>;

    std::cout << "Building index..." << std::endl;
    IndexType index = IndexType(config, &space);
    hnsw::StopWatch sw;
    size_t count = 0;
    const size_t logEvery = config.capacity / 10;
    #pragma omp parallel for
    for (size_t idx = 0; idx < config.capacity; idx++) {
        index.insert(static_cast<int>(idx), idx);
        #pragma omp atomic
        ++count;
        if (count % logEvery == 0) {
            std::cout << "Indexed " << count << " integers -- time elapsed: "
                << sw.elapsed<std::chrono::seconds>() << "s" << std::endl;
        }
    }
    std::cout << "Completed index build -- size: " << index.size() << " -- time elapsed: "
        << sw.elapsed<std::chrono::seconds>() << "s" << std::endl;

    const std::vector<int> queries{10, -10, 7000, 700001, 7000000};

    for (int query : queries) {
        auto ret = index.searchKNN(query, k);
        std::cout << "Querying for " << query << std::endl;
        for (size_t i = 0; i < k; i++) {
            auto [label, dist] = ret[i];
            std::cout << "Rank: " << i << " -- label: " << label
                << ", dist: " << dist << std::endl;
        }
    }
}