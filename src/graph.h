#ifndef GRAPH_H
#define GRAPH_H

#include <mutex>
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
            : adjListsMutexes(config.capacity)
            , enterPoint(INVALID_ID)
            , maxLayer(0)
            , config(config) {
            layerAdjLists.reserve(config.capacity);
        }

        void insert(size_t maxLayer) {
            layerAdjLists.emplace_back(maxLayer + 1);

            // reserve capacities for adjLists
            std::vector<AdjacencyList> & adjLists = layerAdjLists.back();
            adjLists[0].reserve(config.M0_);
            for (size_t layer = 1; layer <= maxLayer; layer++) {
                adjLists[layer].reserve(config.M_);
            }
        }

        void addLink(IdType src, IdType tgt, size_t layer = 0) {
            layerAdjLists[src][layer].push_back(tgt);
        }

        const AdjacencyList & getNeighbours(IdType id, size_t layer = 0) const {
            return layerAdjLists[id][layer];
        }

        void setNeighbours(IdType src, AdjacencyList && tgts, size_t layer = 0) {
            layerAdjLists[src][layer] = std::move(tgts);
        }

        size_t size() const {
            return layerAdjLists.size();
        }

        mutable std::vector<std::mutex> adjListsMutexes;

        IdType enterPoint;
        size_t maxLayer;
        std::mutex maxLayerMutex;

     private:
        const HnswConfig & config;

        std::vector<std::vector<AdjacencyList>> layerAdjLists;
    };
} // namespace  hnsw
#endif // GRAPH_H
