#include <iostream>

#include "embdb/hnsw.h"

int main() {
    hnsw::StopWatch sw;

    using _L2Space = hnsw::L2Space<int>;
    _L2Space space = _L2Space(4);

    using ValueType = _L2Space::ValueType;

    hnsw::HnswConfig config;
    using IndexType = hnsw::HnswIndex<int, std::vector<int>>;
    IndexType index = IndexType(config, &space);

    std::vector<std::pair<size_t, ValueType>> objs({
        {10, {1, 3, 3, 4}},
        {20, {1, 2, 3, 4}},
        {30, {4, 3, 2, 1}},
        {40, {1, 5, 2, 1}},
        {50, {5, 4, 3, 2}},
    });

    ValueType query({5, 3, 3, 2});
    for (const auto & [label, obj] : objs) {
        index.insert(obj, label);
        std::cout << "Label: " << label << " inserted -- distance to query: "
            << space.distance(obj, query) << std::endl;
    }

    std::cout << "Closest item:" <<std::endl;
    auto ret = index.searchKNN(query, 1);
    for (const auto & [label, dist] : ret) {
        std::cout << "Label: " << label << " -- distance to query:"
            << dist << std::endl;
    }

    std::cout << "Three closest items:" <<std::endl;
    ret = index.searchKNN(query, 3);
    for (const auto & [label, dist] : ret) {
        std::cout << "Label: " << label << " -- distance to query:"
            << dist << std::endl;
    }

    std::cout << "Total time elapsed: " << sw.elapsed() << "μs" << std::endl;
    index.checkIntegrity();
}