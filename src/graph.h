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
            : maxGlobalLayer(0)
            , enterPoint(INVALID_ID)
            , maxM(maxM)
            , maxM0(maxM0) {}

        void addNode(IdType id, size_t layer) {
            maxLayer[id] = layer;
            if (enterPoint == INVALID_ID || layer > maxGlobalLayer) {
                enterPoint = id;
                maxGlobalLayer = layer;
            }
        }

        void addLink(IdType src, IdType tgt, size_t layer = 0) {
            neighbours(src, layer).push_back(tgt);
        }

        void addLinks(IdType src, const IdVector & tgts, size_t layer = 0) {
            AdjacencyList & srcAdjList = neighbours(src, layer);
            srcAdjList.insert(srcAdjList.end(), tgts.begin(), tgts.end());
        }

        void setNeighbours(IdType src, AdjacencyList && tgts, size_t layer = 0) {
            AdjacencyList & srcAdjList = neighbours(src, layer);
            srcAdjList = std::forward<AdjacencyList>(tgts);
        }

        const AdjacencyList & getNeighbours(IdType id, size_t layer = 0) const {
            auto it = layerAdjacencyLists.find(id);
            if (it == layerAdjacencyLists.end()) {
                return _DUMMYADJLIST;
            }

            auto & idAdjacencyLists = it->second;
            auto itl = idAdjacencyLists.find(layer);
            if (itl == idAdjacencyLists.end()) {
                return _DUMMYADJLIST;
            }

            return itl->second;
        }

        size_t maxGlobalLayer;
        IdType enterPoint;

     private:
        const size_t maxM;
        const size_t maxM0;

        std::unordered_map<IdType, size_t> maxLayer;
        std::unordered_map<IdType, std::unordered_map<size_t, AdjacencyList>> layerAdjacencyLists;

        AdjacencyList _DUMMYADJLIST;

        AdjacencyList & neighbours(IdType id, size_t layer = 0) {
            if (auto it = maxLayer.find(id); it == maxLayer.end() || it->second < layer) {
                throw std::runtime_error("Invalid layer");
            }

            auto & srcAdjacencyLists = layerAdjacencyLists[id];
            if (auto it = srcAdjacencyLists.find(layer); it != srcAdjacencyLists.end()) {
                return it->second;
            }

            // create and reserve
            AdjacencyList & adjList = srcAdjacencyLists[layer];
            adjList.reserve((layer) ? maxM : maxM0);
            return adjList;
        }
    };
} // namespace  hnsw
#endif // GRAPH_H
