#ifndef GRAPH_H
#define GRAPH_H

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "config.h"

namespace hnsw {
    struct Node {
        Node(size_t maxLayer = 0)
            : maxLayer(maxLayer)
            , deleted(false) {}

        size_t maxLayer;
        bool deleted;

        // edges per level
        std::vector<IdVector> outEdges;
        std::vector<IdVector> inEdges;

        void reserve(const HnswConfig & config) {
            // make sure we hold levels 0, ..., maxLayer
            outEdges.resize(maxLayer + 1);
            inEdges.resize(maxLayer + 1);
            outEdges[0].reserve(config.M0_ + 1);
            inEdges[0].reserve(2 * config.M0_);
            for (size_t layer = 1; layer <= maxLayer; layer++) {
                outEdges[layer].reserve(config.M_ + 1);
                inEdges[layer].reserve(2 * config.M_);
            }
        }

        void reset(size_t maxLayer, const HnswConfig & config) {
            this->maxLayer = maxLayer;
            this->deleted = false;

            reserve(config);

            for (auto & edges : outEdges) {
                edges.clear();
            }
            for (auto & edges : inEdges) {
                edges.clear();
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

        size_t getMaxLayer(IdType id) const {
            return nodes[id].maxLayer;
        }

        bool isDeleted(IdType id) const {
            return nodes[id].deleted;
        }

        void addNode(IdType id, size_t maxLayer) {
            if (id == nodes.size()) {
                nodes.emplace_back(maxLayer);
                // reserve capacities for node
                nodes.back().reserve(config);
            } else {
                if (!isDeleted(id)) {
                    throw std::runtime_error("Corrupt index: overwriting active node.");
                }
                nodes[id].reset(maxLayer, config);
            }
        }

        void addLink(IdType src, IdType tgt, size_t layer) {
            {
                std::unique_lock tgtInLock(inMutexes[tgt]);
                nodes[tgt].inEdges[layer].push_back(src);
            }
            nodes[src].outEdges[layer].push_back(tgt);
        }

        void setNeighbours(IdType src, IdVector && tgts, size_t layer,
                           bool checkTgtDeleted = false) {
            for (IdType prevTgt : nodes[src].outEdges[layer]) {
                std::unique_lock prevInLock(inMutexes[prevTgt]);
                IdVector & prevIns = nodes[prevTgt].inEdges[layer];
                if (!isDeleted(prevTgt)) {
                    auto it = std::find(prevIns.begin(), prevIns.end(), src);
                    *it = std::move(prevIns.back());
                    prevIns.pop_back();
                }
            }

            for (IdType tgt : tgts) {
                std::unique_lock currInLock(inMutexes[tgt]);
                if (!isDeleted(tgt)) {
                    nodes[tgt].inEdges[layer].push_back(src);
                }
            }

            IdVector & out = nodes[src].outEdges[layer];
            out = std::move(tgts);
            if (checkTgtDeleted) {
                // remove deleted neighbours
                auto it = std::remove_if(out.begin(), out.end(),
                                         [this](IdType id) { return isDeleted(id); });
                out.erase(it, out.end());
            }
        }

        void removeNode(IdType id) {
            Node & node = nodes[id];
            {
                std::unique_lock outLock(outMutexes[id]);
                std::unique_lock inLock(inMutexes[id]);
                node.deleted = true;
            }

            for (size_t layer = 0; layer <= node.maxLayer; layer++) {
                for (IdType src : node.inEdges[layer]) {
                    std::unique_lock lock(outMutexes[src]);
                    IdVector & srcOuts = nodes[src].outEdges[layer];
                    auto it = std::find(srcOuts.begin(), srcOuts.end(), id);
                    if (it != srcOuts.end()) {
                        *it = std::move(srcOuts.back());
                        srcOuts.pop_back();
                    }
                }

                for (IdType tgt : node.outEdges[layer]) {
                    std::unique_lock lock(inMutexes[tgt]);
                    IdVector & tgtIns = nodes[tgt].inEdges[layer];
                    auto it = std::find(tgtIns.begin(), tgtIns.end(), id);
                    if (it != tgtIns.end()) {
                        *it = std::move(tgtIns.back());
                        tgtIns.pop_back();
                    }
                }
            }
        }

        void refreshEnterPoint() {
            // TODO: Is this good enough?
            if (enterPoint == INVALID_ID || !nodes[enterPoint].deleted) {
                return;
            }

            enterPoint = INVALID_ID;
            maxLayer = 0;
            size_t maxOuts = 0;
            for (size_t id = 0; id < nodes.size(); id++) {
                Node & node = nodes[id];
                if (node.deleted) {
                    continue;
                }

                if (enterPoint == INVALID_ID || maxLayer < node.maxLayer ||
                    (maxLayer == node.maxLayer && maxOuts < node.outEdges[maxLayer].size())) {
                    enterPoint = id;
                    maxLayer = node.maxLayer;
                    maxOuts = node.outEdges[maxLayer].size();
                }
            }
        }

        void checkIntegrity(const std::unordered_set<IdType> & deleted) const {
            // not thread-safe with inserting/modifying
            if (nodes.size() >= 1UL << 32) {
                throw std::runtime_error("Index is too large; cannot check integrity");
            }

            size_t _deleted = 0, valid = 0;
            std::vector<std::unordered_map<size_t, uint8_t>> edges(maxLayer + 1);
            for (size_t id = 0; id < nodes.size(); id++) {
                const Node & node = nodes.at(id);
                if (node.deleted) {
                    if (deleted.find(id) != deleted.end()) {
                        _deleted++;
                        continue;
                    }

                    throw std::runtime_error("Valid id: " + std::to_string(id) +
                                             " deleted from graph");
                }

                valid++;

                for (size_t layer = 0; layer <= node.maxLayer; layer++) {
                    if (node.outEdges[layer].size() > ((layer > 0) ? config.M_ : config.M0_)) {
                        throw std::runtime_error("Invalid out size: " + std::to_string(id) +
                                                 " in layer: " + std::to_string(layer));
                    }
                    for (const IdType tgt : node.outEdges[layer]) {
                        if (nodes[tgt].deleted || nodes[tgt].maxLayer < layer) {
                            throw std::runtime_error("Unexpected node: " + std::to_string(tgt) +
                                                     " in layer: " + std::to_string(layer));
                        }
                        edges[layer][(id << 32) | tgt] |= 1;
                    }

                    for (const IdType src : node.inEdges[layer]) {
                        if (nodes[src].deleted || nodes[src].maxLayer < layer) {
                            throw std::runtime_error("Unexpected node: " + std::to_string(src) +
                                                     " in layer: " + std::to_string(layer));
                        }
                        edges[layer][(src << 32) | id] |= 2;
                    }
                }
            }

            if (_deleted != deleted.size()) {
                throw std::runtime_error("Discrepancy between deleted in graph and space");
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
