#include <iostream>

#include "hnsw.h"

int main() {
    using _L2Space = hnsw::L2Space<int>;
    _L2Space space = _L2Space(4);

    using ValueType = _L2Space::ValueType;

    hnsw::HnswConfig config;
    using IndexType = hnsw::HnswIndex<int, std::vector<int>>;
    IndexType index = IndexType(config, space);

    std::unordered_map<size_t, ValueType> objs({
        {10, {1, 3, 3, 4}},
        {20, {1, 2, 3, 4}},
        {30, {4, 3, 2, 1}}
    });
    ValueType query({5, 3, 3, 2});
    for (const auto & [label, obj] : objs) {
        std::cout << space.distance(obj, query) << std::endl;
        hnsw::IdType id = index.insert(obj, label);
        std::cout << "Label: " << label << " inserted -- id: " << id  << std::endl;
    }

    auto ret = index.searchKNN(query, 1);
    for (const auto & [id, dist] : ret) {
        std::cout << id << " -- " << dist << std::endl;
    }

    ret = index.searchKNN(query, 2);
    for (const auto & [id, dist] : ret) {
        std::cout << id << " -- " << dist << std::endl;
    }
}