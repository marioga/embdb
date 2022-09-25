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
    hnsw::IdType id2 = index.addDebug(obj2, 1);
    hnsw::IdType id3 = index.addDebug(obj3, 2);
    index.addLinkDebug(id2, id3, 1);

    std::unordered_set<hnsw::IdType> temp({id2});
    auto queue = index.searchLayer(obj1, temp, 1, 1);
    while (queue.size()) {
        std::cout << queue.top().first << " -- " << queue.top().second << std::endl;
        queue.pop();
    }
}