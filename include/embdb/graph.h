#ifndef GRAPH_H
#define GRAPH_H

#include <mutex>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "config.h"

namespace hnsw {
    struct Node {
        Node(IdType id = INVALID_ID, size_t maxLayer = 0)
            : id(id)
           , maxLayer(maxLayer) {}

        IdType id;
        size_t maxLayer;

        // edges per level
        std::vector<IdVector> outEdges;
        std::vector<IdVector> inEdges;

        void reserve(const HnswConfig & config) {
            // make sure we hold levels 0, ..., maxLayer
            outEdges.resize(maxLayer + 1);
            inEdges.resize(maxLayer + 1);
            outEdges[0].reserve(config.M0_ + 1);
            inEdges[0].reserve(config.M0_ + 1);
            for (size_t layer = 1; layer <= maxLayer; layer++) {
                outEdges[layer].reserve(config.M_ + 1);
                inEdges[layer].reserve(config.M_ + 1);
            }
        }
    };

    class HNSWGraph {
     public:
        HNSWGraph(const HnswConfig & config)
            : enterPoint(INVALID_ID)
            , maxLayer(0)
            , config(config)
            , inMutexes(config.capacity)
            , outMutexes(config.capacity) {
            nodes.reserve(config.capacity);
        }

        const IdVector & getNeighbours(IdType src, size_t layer) const {
            return nodes[src].outEdges[layer];
        }

        void addNode(size_t maxLayer) {
            nodes.emplace_back(nodes.size(), maxLayer);
            // reserve capacities for node
            nodes.back().reserve(config);
        }

        void addLink(IdType src, IdType tgt, size_t layer) {
            {
                std::unique_lock<std::mutex> tgtInLock(inMutexes[tgt]);
                nodes[tgt].inEdges[layer].push_back(src);
            }
            nodes[src].outEdges[layer].push_back(tgt);
        }

        void setNeighbours(IdType src, IdVector && tgts, size_t layer) {
            for (IdType prevTgt : nodes[src].outEdges[layer]) {
                std::unique_lock<std::mutex> prevInlock(inMutexes[prevTgt]);
                IdVector & prevIns = nodes[prevTgt].inEdges[layer];
                auto it = std::find(prevIns.begin(), prevIns.end(), src);
                *it = std::move(prevIns.back());
                prevIns.pop_back();
            }
            for (IdType tgt : tgts) {
                std::unique_lock<std::mutex> currInLock(inMutexes[tgt]);
                nodes[tgt].inEdges[layer].push_back(src);
            }
            nodes[src].outEdges[layer] = std::move(tgts);
        }

        void checkIntegrity() const {
            // not thread-safe with inserting/modifying
            IdType size = nodes.size();
            if (size >= 1UL << 32) {
                throw std::runtime_error("Index is too large; cannot check integrity");
            }

            std::vector<std::unordered_map<size_t, uint8_t>> edges(maxLayer + 1);
            for (size_t id = 0; id < size; id++) {
                const Node & node = nodes.at(id);
                if (node.id != id) {
                    throw std::runtime_error("Invalid id: " + std::to_string(id));
                }
                for (size_t layer = 0; layer <= node.maxLayer; layer++) {
                    if (node.outEdges[layer].size() > ((layer > 0) ? config.M_ : config.M0_)) {
                        throw std::runtime_error("Invalid out size: " + std::to_string(node.id) +
                                                 " in layer: " + std::to_string(layer));
                    }
                    for (const IdType tgt : node.outEdges[layer]) {
                        if (nodes[tgt].maxLayer < layer) {
                            throw std::runtime_error("Unexpected node: " + std::to_string(tgt) +
                                                     " in layer: " + std::to_string(layer));
                        }
                        edges[layer][(node.id << 32) | tgt] |= 1;
                    }

                    for (const IdType src : node.inEdges[layer]) {
                        if (nodes[src].maxLayer < layer) {
                            throw std::runtime_error("Unexpected node: " + std::to_string(src) +
                                                     " in layer: " + std::to_string(layer));
                        }
                        edges[layer][(src << 32) | node.id] |= 2;
                    }
                }
            }

            for (size_t layer = 0; layer < edges.size(); layer++) {
                const auto & layerEdges = edges[layer];
                for (const auto & [key, flag] : layerEdges) {
                    IdType tgt = key & ((1UL << 32) - 1);
                    IdType src = key >> 32;
                    if (flag != 3) {
                        throw std::runtime_error("Incomplete edge: " + std::to_string(src) +
                                                 " -> " + std::to_string(tgt) + " in layer: " +
                                                 std::to_string(layer));
                    }
                }
            }
        }

        std::mutex * getMutex(IdType id) const {
            return &outMutexes[id];
        }

        std::mutex * getMaxLayerMutex() const {
            return &maxLayerMutex;
        }

        IdType enterPoint;
        size_t maxLayer;

     private:
        const HnswConfig & config;

        std::vector<Node> nodes;

        mutable std::mutex maxLayerMutex;
        mutable std::vector<std::mutex> inMutexes;
        mutable std::vector<std::mutex> outMutexes;
    };
} // namespace hnsw
#endif // GRAPH_H
