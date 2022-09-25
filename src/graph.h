#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_map>
#include <vector>

namespace hnsw {
    using IdType = size_t;
    using AdjacencyList = std::vector<IdType>;

    class HNSWGraph {
     public:
        HNSWGraph(size_t maxM, size_t maxM0)
            : maxM(maxM)
            , maxM0(maxM0)
            , maxGlobalLevel(0) {}

        void addNode(IdType id, size_t level) {
            maxLevel[id] = level;
            maxGlobalLevel = std::max(level, maxGlobalLevel);
        }

        void addLink(IdType src, IdType tgt, size_t level = 0) {
            if (auto it = maxLevel.find(src); it == maxLevel.end() || it->second < level) {
                throw std::runtime_error("Invalid level");
            }

            AdjacencyList * adjList;
            auto & srcAdjacencyLists = levelAdjacencyLists[src];
            if (auto it = srcAdjacencyLists.find(level); it == srcAdjacencyLists.end()) {
                // create and reserve
                adjList = &srcAdjacencyLists[level];
                adjList->reserve((level) ? maxM : maxM0);
            } else {
                adjList = &it->second;
            }

            adjList->push_back(tgt);
        }

        const AdjacencyList & neighbours(IdType id, size_t level = 0) const {
            auto it = levelAdjacencyLists.find(id);
            if (it == levelAdjacencyLists.end()) {
                return _DUMMYADJLIST;
            }

            auto & idAdjacencyLists = it->second;
            auto itl = idAdjacencyLists.find(level);
            if (itl == idAdjacencyLists.end()) {
                return _DUMMYADJLIST;
            }

            return itl->second;
        }

     private:
        const size_t maxM;
        const size_t maxM0;

        size_t maxGlobalLevel;

        std::unordered_map<IdType, size_t> maxLevel;
        std::unordered_map<IdType, std::unordered_map<size_t, AdjacencyList>> levelAdjacencyLists;

        AdjacencyList _DUMMYADJLIST;
    };
} // namespace  hnsw
#endif // GRAPH_H
