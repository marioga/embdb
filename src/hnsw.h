#ifndef HNSW_H
#define HNSW_H

#include <algorithm>
#include <functional>
#include <random>
#include <unordered_set>
#include <vector>

#include "graph.h"
#include "space.h"

namespace hnsw {
    struct HnswConfig {
        size_t capacity = 128;

        size_t efConstruction = 100;
        size_t M_ = 16;
        size_t M0_ = 32;

        std::optional<std::mt19937::result_type> levelGenSeed = std::nullopt;
        // rate of level exponential distribution; corresponds to 1/mL for mL as in the paper
        double levelExpLambda = log(static_cast<double>(M_));

        bool extendCandidates = false;
        bool keepPrunedConnections = false;
    };

    template<typename dist_t, typename ValueType, typename LabelType = size_t>
    class HnswIndex {
     public:
        using SpaceType = MetricSpace<dist_t, ValueType>;

        using NNPair = std::pair<IdType, dist_t>;
        using NNVector = std::vector<NNPair>;

        HnswIndex(const HnswConfig & config, const SpaceType & space)
            : config(config)
            , space(space)
            , size(0)
            , capacity(config.capacity)
            , graph(config.M_, config.M0_) {
            elements.reserve(capacity);
            labels.reserve(capacity);

            if (config.levelGenSeed.has_value()) {
                levelGen.seed(config.levelGenSeed.value());
            } else {
                std::random_device rd;
                levelGen.seed(rd());
            }
        }

        IdType insert(const ValueType & value, LabelType label) {
            if (auto it = labelsInv.find(label); it != labelsInv.end()) {
                // TODO: implement updates
                throw std::runtime_error("Label already present");
            }

            if (size >= capacity) {
                throw std::runtime_error("Capacity reached -- index must be resized");
            }

            IdType id = size++;
            elements.push_back(value);
            labels.push_back(label);
            labelsInv[label] = id;

            // determine level by sampling from levelExpDist
            std::exponential_distribution<double> levelExpDist(config.levelExpLambda);
            size_t insertLevel = static_cast<size_t>(std::floor(levelExpDist(levelGen)));

            const size_t prevMaxLevel = graph.maxGlobalLevel;
            IdType currEnterPoint = graph.enterPoint;

            // add node to hnsw graph (might change enterPoint/maxGlobalLevel)
            graph.addNode(id, insertLevel);

            if (currEnterPoint == INVALID_ID) {
                // first point ever added; nothing left to do
                return id;
            }

            for (size_t level = prevMaxLevel; level > insertLevel; level--) {
                NNVector found = searchLayer(value, currEnterPoint, 1, level);
                currEnterPoint = found.front().first;
            }

            std::unordered_set<IdType> enterPoints({currEnterPoint});
            size_t level = std::min(insertLevel, prevMaxLevel);
            do {
                NNVector candidates = searchLayer(value, enterPoints, config.efConstruction, level);
                size_t M = (level > 0) ? config.M_ : config.M0_;
                IdVector neighbours = selectNeighboursHeuristic(value, candidates, M, level);
                addBidirectionalLinks(id, neighbours, M, level, value);

                // reset enterPoints to candidates
                enterPoints.clear();
                std::transform(candidates.begin(), candidates.end(),
                               std::inserter(enterPoints, enterPoints.begin()),
                               [](const auto & p) { return p.first; });
            }
            while (level-- > 0);

            return id;
        }

        NNVector searchKNN(const ValueType & query, size_t k, size_t ef) const {
            IdType currEnterPoint = graph.enterPoint;
            for (size_t level = graph.maxGlobalLevel; level > 0; level--) {
                NNVector found = searchLayer(query, currEnterPoint, 1, level);
                currEnterPoint = found.front().first;
            }
            NNVector found = searchLayer(query, currEnterPoint, ef);

            NNVector ret(std::min(k, found.size()));
            std::partial_sort_copy(found.begin(), found.end(), ret.begin(), ret.end(),
                                   maxHeapCompare);
            return ret;
        }

        IdType addDebug(const ValueType & entry, size_t level) {
            elements.push_back(entry);
            IdType id = size++;
            graph.addNode(id, level);
            return id;
        }

        void addLinkDebug(IdType src, IdType tgt, size_t level) {
            graph.addLink(src, tgt, level);
        }

     protected:
        template<typename Compare = std::greater<dist_t>>
        struct CompareNNPair {
            Compare cmp;

            bool operator() (const NNPair & a, const NNPair & b) const {
                return cmp(a.second, b.second);
            }
        };
        CompareNNPair<> minHeapCompare;
        CompareNNPair<std::less<dist_t>> maxHeapCompare;

        NNVector searchLayer(const ValueType & query, IdType enterPoint,
                             size_t ef, size_t layer = 0) const {
            std::unordered_set<IdType> visited({enterPoint});

            NNPair entry(enterPoint, distance(query, enterPoint));
            NNVector candidates({entry});
            NNVector found({entry});

            return searchLayerImpl(query, candidates, found, visited, ef, layer);
        }

        NNVector searchLayer(const ValueType & query,
                             const std::unordered_set<IdType> & enterPoints,
                             size_t ef, size_t layer = 0) const {
            std::unordered_set<IdType> visited(enterPoints);

            NNVector candidates, found;
            candidates.reserve(enterPoints.size());
            found.reserve(enterPoints.size());
            for (IdType pid : enterPoints) {
                dist_t dist = distance(query, pid);
                candidates.emplace_back(pid, dist);
                found.emplace_back(pid, dist);
            }

            return searchLayerImpl(query, candidates, found, visited, ef, layer);
        }


        NNVector searchLayerImpl(const ValueType & query, NNVector & candidates,
                                 NNVector & found, std::unordered_set<IdType> & visited,
                                 size_t ef, size_t layer = 0) const {
            std::make_heap(candidates.begin(), candidates.end(), minHeapCompare);
            std::make_heap(found.begin(), found.end(), maxHeapCompare);

            while (!candidates.empty()) {
                // closest candidate to query
                auto [cid, cdist] = candidates.front();
                // farthest found point to query
                auto fdist = found.front().second;
                if (cdist > fdist) {
                    // we are done here
                    break;
                }

                // pop current candidate
                std::pop_heap(candidates.begin(), candidates.end(), minHeapCompare);
                candidates.pop_back();

                for (IdType nid : graph.getNeighbours(cid, layer)) {
                    auto it = visited.insert(nid);
                    if (!it.second) {
                        // already visited
                        continue;
                    }

                    dist_t ndist = distance(query, nid);
                    if (ndist < fdist or found.size() < ef) {
                        candidates.emplace_back(nid, ndist);
                        std::push_heap(candidates.begin(), candidates.end(), minHeapCompare);
                        found.emplace_back(nid, ndist);
                        std::push_heap(found.begin(), found.end(), maxHeapCompare);
                        if (found.size() > ef) {
                            std::pop_heap(found.begin(), found.end(), maxHeapCompare);
                            found.pop_back();
                        }
                    }

                    // update farthest point
                    fdist = found.front().second;
                }
            }

            return found;
        }

        IdVector selectNeighboursHeuristic(const ValueType & query, const NNVector & candidates,
                                           size_t M, size_t layer) const {
            NNVector candidatesMinHeap(candidates);
            if (config.extendCandidates) {
                std::unordered_set<IdType> visited;
                std::transform(candidates.begin(), candidates.end(),
                               std::inserter(visited, visited.begin()),
                               [](const auto & p) { return p.first; });

                for (const NNPair & c : candidates) {
                    for (IdType nid : graph.getNeighbours(c.first, layer)) {
                        if (auto it = visited.find(nid); it == visited.end()) {
                            visited.insert(nid);
                            dist_t ndist = distance(query, nid);
                            candidatesMinHeap.emplace_back(nid, ndist);
                        }
                    }
                }
            }

            IdVector ret;
            if (candidatesMinHeap.size() < M) {
                // no heuristic necessary
                ret.reserve(candidatesMinHeap.size());
                for (const NNPair & c : candidatesMinHeap) {
                    ret.push_back(c.first);
                }
                return ret;
            }

            ret.reserve(M);
            std::make_heap(candidatesMinHeap.begin(), candidatesMinHeap.end(), minHeapCompare);
            IdVector pruned;
            while (!candidatesMinHeap.empty() && ret.size() < M) {
                auto [cid, cdist] = candidatesMinHeap.front();
                std::pop_heap(candidatesMinHeap.begin(), candidatesMinHeap.end(), minHeapCompare);
                candidatesMinHeap.pop_back();

                bool include = true;
                const ValueType & cvalue = (*this)[cid];
                // heuristic: include only points that are closer to query element
                // than to any other previously returned elements
                for (IdType rid : ret) {
                    dist_t dist = distance(cvalue, rid);
                    if (dist < cdist) {
                        include = false;
                        break;
                    }
                }

                if (include) {
                    ret.push_back(cid);
                } else if (config.keepPrunedConnections) {
                    pruned.push_back(cid);
                }
            }

            if (config.keepPrunedConnections && ret.size() < M) {
                for (IdType pid : pruned) {
                    // pruned is already sorted by proximity
                    ret.push_back(pid);
                    if (ret.size() == M) {
                        break;
                    }
                }
            }

            return ret;
        }

        void addBidirectionalLinks(IdType src, const IdVector & tgts, size_t M, size_t layer,
                                   const ValueType & query) {
            graph.addLinks(src, tgts, layer);
            for (IdType tgt : tgts) {
                graph.addLink(tgt, src, layer);

                // shrink connections if necessary
                const AdjacencyList & tgtNeighbours = graph.getNeighbours(tgt, layer);
                if (tgtNeighbours.size() > M) {
                    NNVector tgtCandidates;
                    tgtCandidates.reserve(tgtNeighbours.size());
                    for (IdType nid : tgtNeighbours) {
                        tgtCandidates.emplace_back(nid, distance(query, nid));
                    }
                    IdVector newTgtNeighbours = selectNeighboursHeuristic(query, tgtCandidates,
                                                                          M, layer);
                    graph.setNeighbours(tgt, std::move(newTgtNeighbours), layer);
                }
            }
        }

        dist_t distance(const ValueType & query, const ValueType & elem) const {
            return space.distance(query, elem);
        }

        dist_t distance(const ValueType & query, IdType id) const {
            return distance(query, (*this)[id]);
        }

        dist_t distance(IdType id1, IdType id2) const {
            return distance((*this)[id1], (*this)[id2]);
        }

        ValueType & operator[](IdType id) {
            return elements[id];
        }

        const ValueType & operator[](IdType id) const {
            return elements[id];
        }

     private:
        const HnswConfig & config;
        const SpaceType & space;

        std::vector<ValueType> elements;
        std::vector<LabelType> labels;
        std::unordered_map<LabelType, IdType> labelsInv;
        size_t size;
        size_t capacity;

        HNSWGraph graph;

        std::mt19937 levelGen;
    };
} // namespace  hnsw
#endif // HNSW_H