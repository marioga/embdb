#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_map>
#include <vector>

#include "config.h"

namespace hnsw {
    using IdType = size_t;
    using IdVector = std::vector<IdType>;
    using AdjacencyList = IdVector;

    const IdType INVALID_ID = static_cast<IdType>(-1);

    class HNSWGraph {
     public:
        HNSWGraph(const HnswConfig & config)
            : enterPoint(INVALID_ID)
            , maxLayer(0)
            , config(config) {
            layerAdjacencyLists.reserve(config.capacity);
        }

        void insert(size_t maxLayer) {
            layerAdjacencyLists.emplace_back(maxLayer + 1);
            std::vector<AdjacencyList> & adjLists = layerAdjacencyLists.back();
            adjLists[0].reserve(config.M0_);
            for (size_t layer = 1; layer <= maxLayer; layer++) {
                adjLists[layer].reserve(config.M_);
            }
        }

        void addLink(IdType src, IdType tgt, size_t layer = 0) {
            layerAdjacencyLists[src][layer].push_back(tgt);
        }

        const AdjacencyList & getNeighbours(IdType id, size_t layer = 0) const {
            return layerAdjacencyLists[id][layer];
        }

        void setNeighbours(IdType src, AdjacencyList && tgts, size_t layer = 0) {
            layerAdjacencyLists[src][layer] = std::move(tgts);
        }

        void setNeighbours(IdType src, const AdjacencyList & tgts, size_t layer = 0) {
            layerAdjacencyLists[src][layer] = tgts;
        }

        size_t size() const {
            return layerAdjacencyLists.size();
        }

        IdType enterPoint;
        size_t maxLayer;

     private:
        const HnswConfig & config;

        std::vector<std::vector<AdjacencyList>> layerAdjacencyLists;
    };
} // namespace  hnsw
#endif // GRAPH_H
