#include <iostream>
#include "omp.h"

#include "embdb/hnsw.h"

class IntegerSpace : public hnsw::MetricSpace<int, int> {
 public:
    int distance(const int & obj1, const int & obj2) const override {
        return abs(obj1 - obj2);
    }
};

int main() {
    const size_t k = 5;

    IntegerSpace space;

    hnsw::HnswConfig config;
    config.capacity = 1000000;
    config.efConstruction = 64;
    config.ef = 64;

    using IndexType = hnsw::HnswIndex<int, int>;

    const size_t logEvery = config.capacity / 10;

    std::cout << "Building index (first half)..." << std::endl;
    IndexType index = IndexType(config, &space);
    size_t count = 0;
    hnsw::StopWatch sw;
    // omp_set_num_threads(2);
    #pragma omp parallel for
    for (size_t idx = 0; idx < config.capacity / 2; idx++) {
        index.insert(static_cast<int>(idx), idx);
        #pragma omp atomic
        ++count;
        if (count % logEvery == 0) {
            std::cout << "Indexed " << count << " integers -- time elapsed: "
                << sw.elapsed<std::chrono::milliseconds>() << "ms" << std::endl;
        }
    }
    std::cout << "Completed index build (first half) -- size: " << index.size()
        << " -- time elapsed: " << sw.elapsed<std::chrono::milliseconds>() << "ms" << std::endl;

    index.checkIntegrity();

    std::cout << "Removing first half even labels..." << std::endl;
    count = 0;
    sw.reset();
    #pragma omp parallel for
    for (size_t idx = 0; idx < config.capacity / 2; idx += 2) {
        index.remove(idx);
        #pragma omp atomic
        ++count;
        if (count % logEvery == 0) {
            std::cout << "Removed " << count << " integers -- time elapsed: "
                << sw.elapsed<std::chrono::milliseconds>() << "ms" << std::endl;
        }
    }
    std::cout << "Completed removal -- size: " << index.size()
        << " -- time elapsed: " << sw.elapsed<std::chrono::milliseconds>() << "ms" << std::endl;

    index.checkIntegrity();

    std::cout << "Building index (second half)..." << std::endl;
    count = 0;
    sw.reset();
    #pragma omp parallel for
    for (size_t idx = config.capacity / 2; idx < config.capacity; idx++) {
        // if ((idx - config.capacity / 2) % 2 == 0) {
        //     index.remove(idx - config.capacity / 2);
        // }
        index.insert(static_cast<int>(idx), idx);
        #pragma omp atomic
        ++count;
        if (count % logEvery == 0) {
            std::cout << "Inserted " << count << " integers -- time elapsed: "
                << sw.elapsed<std::chrono::milliseconds>() << "ms" << std::endl;
        }
    }
    std::cout << "Completed index build (second half) -- size: " << index.size()
        << " -- time elapsed: " << sw.elapsed<std::chrono::milliseconds>() << "ms" << std::endl;

    index.checkIntegrity();

    const int midPoint = static_cast<int>(config.capacity / 2);
    const std::vector<int> queries{-333, 0, 101, midPoint, 703071, 3040446, 7345678};

    for (int query : queries) {
        auto ret = index.searchKNN(query, k);
        std::cout << "Querying for label: " << query << std::endl;
        for (size_t i = 0; i < ret.size(); i++) {
            auto [label, dist] = ret[i];
            std::cout << "Rank: " << i << " -- label: " << label
                << ", dist: " << dist << std::endl;
        }
    }
}