#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_map>
#include <vector>

namespace hnsw {
    using IdType = size_t;
    using IdVector = std::vector<IdType>;
    using AdjacencyList = IdVector;

    const IdType INVALID_ID = static_cast<IdType>(-1);

    class HNSWGraph {
     public:
        HNSWGraph(size_t maxM, size_t maxM0)
            : maxGlobalLevel(0)
            , enterPoint(INVALID_ID)
            , maxM(maxM)
            , maxM0(maxM0) {}

        void addNode(IdType id, size_t level) {
            maxLevel[id] = level;
            if (enterPoint == INVALID_ID || level > maxGlobalLevel) {
                enterPoint = id;
                maxGlobalLevel = level;
            }
        }

        void addLink(IdType src, IdType tgt, size_t level = 0) {
            neighbours(src, level).push_back(tgt);
        }

        void addLinks(IdType src, const IdVector & tgts, size_t level = 0) {
            AdjacencyList & srcAdjList = neighbours(src, level);
            srcAdjList.insert(srcAdjList.end(), tgts.begin(), tgts.end());
        }

        void setNeighbours(IdType src, AdjacencyList && tgts, size_t level = 0) {
            AdjacencyList & srcAdjList = neighbours(src, level);
            srcAdjList = std::forward<AdjacencyList>(tgts);
        }

        const AdjacencyList & getNeighbours(IdType id, size_t level = 0) const {
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

        size_t maxGlobalLevel;
        IdType enterPoint;

     private:
        const size_t maxM;
        const size_t maxM0;

        std::unordered_map<IdType, size_t> maxLevel;
        std::unordered_map<IdType, std::unordered_map<size_t, AdjacencyList>> levelAdjacencyLists;

        AdjacencyList _DUMMYADJLIST;

        AdjacencyList & neighbours(IdType id, size_t level = 0) {
            if (auto it = maxLevel.find(id); it == maxLevel.end() || it->second < level) {
                throw std::runtime_error("Invalid level");
            }

            auto & srcAdjacencyLists = levelAdjacencyLists[id];
            if (auto it = srcAdjacencyLists.find(level); it != srcAdjacencyLists.end()) {
                return it->second;
            }

            // create and reserve
            AdjacencyList & adjList = srcAdjacencyLists[level];
            adjList.reserve((level) ? maxM : maxM0);
            return adjList;
        }
    };
} // namespace  hnsw
#endif // GRAPH_H
