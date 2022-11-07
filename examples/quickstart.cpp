#include <iostream>

#include "embdb/hnsw.h"

int main() {
    hnsw::initLogging();

    hnsw::StopWatch sw;

    using _L2Space = hnsw::L2Space<int>;
    _L2Space space = _L2Space(4);

    using ValueType = _L2Space::ValueType;

    hnsw::HnswConfig config;
    using IndexType = hnsw::HnswIndex<int, std::vector<int>>;
    IndexType index = IndexType(config, &space);

    const std::vector<std::pair<size_t, ValueType>> objs({
        {10, {1, 3, 3, 4}},
        {20, {1, 2, 3, 4}},
        {30, {4, 3, 2, 1}},
        {40, {1, 5, 2, 1}},
        {50, {5, 4, 3, 2}},
    });

    ValueType query({5, 3, 3, 2});
    for (const auto & [label, obj] : objs) {
        index.insert(obj, label);
        SPDLOG_INFO("Label: {} inserted -- distance to query: {}",
                    label, space.distance(obj, query));
    }

    SPDLOG_INFO("Three closest items: ");
    auto ret = index.searchKNN(query, 3);
    for (const auto & [label, dist] : ret) {
        SPDLOG_INFO("Label: {} inserted -- distance to query: {}", label, dist);
    }
    SPDLOG_INFO(std::vector<int>({3, 4, 5}));

    index.checkIntegrity();

    index.remove(10);
    index.remove(20);
    index.remove(50);

    index.checkIntegrity();

    SPDLOG_INFO("Three closest items after deletion: ");
    ret = index.searchKNN(query, 3);
    for (const auto & [label, dist] : ret) {
        SPDLOG_INFO("Label: {} inserted -- distance to query: {}", label, dist);
    }

    const std::vector<std::pair<size_t, ValueType>> newObjs({
        {30, {7, 7, 7, 7}},
        {60, {5, 3, 2, 2}},
        {70, {5, 2, 2, 5}},
    });

    for (const auto & [label, obj] : newObjs) {
        index.remove(label);
        index.insert(obj, label);
        SPDLOG_INFO("Label: {} inserted -- distance to query: {}",
                    label, space.distance(obj, query));
    }

    index.checkIntegrity();

    SPDLOG_INFO("Three closest items after re-insertion: ");
    ret = index.searchKNN(query, 3);
    for (const auto & [label, dist] : ret) {
        SPDLOG_INFO("Label: {} inserted -- distance to query: {}", label, dist);
    }

    SPDLOG_INFO("Total time elapsed: {}μs", sw.elapsed());
}