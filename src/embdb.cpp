#include <iostream>

#include "hnsw.h"

int main() {
    using _L2Space = hnsw::L2Space<int>;
    _L2Space space = _L2Space(4);

    using ValueType = _L2Space::ValueType;

    ValueType obj1({1, 2, 3, 4});
    ValueType obj2({4, 3, 2, 1});
    ValueType obj3({1, 2, 1, 2});
    std::cout << space.distance(obj1, obj2) << std::endl;
    std::cout << space.distance(obj1, obj3) << std::endl;

    hnsw::HnswConfig config;
    using IndexType = hnsw::HnswIndex<int, std::vector<int>>;
    IndexType index = IndexType(config, space);
    hnsw::IdType id2 = index.insert(obj2, 20);
    std::cout << id2 << " inserted" << std::endl;
    hnsw::IdType id3 = index.insert(obj3, 30);
    std::cout << id3 << " inserted" << std::endl;

    auto ret = index.searchKNN(obj1, 1, 8);
    for (const auto & [id, dist] : ret) {
        std::cout << id << " -- " << dist << std::endl;
    }
}