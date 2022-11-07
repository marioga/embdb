#ifndef HNSW_H
#define HNSW_H

#include <algorithm>
#include <functional>
#include <mutex>
#include <random>
#include <unordered_set>
#include <vector>

#include "graph.h"
#include "space.h"

namespace hnsw {
    template<typename dist_t, typename ValueType, typename LabelType = size_t>
    class HnswIndex {
     protected:
        using SpaceType = MetricSpace<dist_t, ValueType, LabelType>;

        using NNPair = std::pair<IdType, dist_t>;
        using NNVector = std::vector<NNPair>;

     public:
        HnswIndex(const HnswConfig & config, SpaceType * space)
            : config(config)
            , graph(config)
            , space(space) {
            space->setCapacity(config.capacity);

            if (config.layerGenSeed.has_value()) {
                layerGen.seed(config.layerGenSeed.value());
            } else {
                std::random_device rd;
                layerGen.seed(rd());
            }
        }

        size_t size() const {
            std::unique_lock insertLock(insertMutex);
            return space->size();
        }

        void checkIntegrity() const {
            std::unordered_set<IdType> deleted = space->checkIntegrity();
            graph.checkIntegrity(deleted);
        }

        void insert(const ValueType & value, LabelType label) {
            IdType id;
            size_t maxLayer;
            {
                std::unique_lock insertLock(insertMutex);
                id = space->add(value, label);
                if (id == INVALID_ID) {
                    insertLock.unlock();
                    throw std::runtime_error("Label already present");
                }

                // determine top insertion layer by sampling from layerExpDist
                std::exponential_distribution<double> layerExpDist(config.layerExpLambda);
                maxLayer = static_cast<size_t>(std::floor(layerExpDist(layerGen)));

                graph.addNode(id, maxLayer);
            }

            std::unique_lock maxLayerLock(*graph.getMaxLayerMutex());

            IdType currEnterPoint = graph.enterPoint;
            if (currEnterPoint == INVALID_ID) {
                // first point ever added
                graph.enterPoint = id;
                graph.maxLayer = maxLayer;
                return;
            }

            if (maxLayer <= graph.maxLayer) {
                maxLayerLock.unlock();
            }

            std::unique_lock idLock(*graph.getMutex(id));

            size_t layer = graph.maxLayer;
            while (layer > maxLayer) {
                NNVector candidates = searchLayer(value, currEnterPoint, 1, layer);
                currEnterPoint = candidates.front().first;
                layer--;
            }

            std::unordered_set<IdType> enterPoints({currEnterPoint});
            while (true) {
                NNVector candidates = searchLayer(value, enterPoints, config.efConstruction, layer);
                const size_t M = (layer > 0) ? config.M_ : config.M0_;
                IdVector neighbours = selectNeighboursHeuristic(value, candidates, M, layer,
                                                                config.extendCandidates,
                                                                config.keepPrunedConnections);
                // add bidirectional links
                for (IdType tgt : neighbours) {
                    std::unique_lock tgtLock(*graph.getMutex(tgt));
                    graph.addLink(tgt, id, layer);
                    // shrink connections if necessary
                    const IdVector & tgtNeighbours = graph.getNeighbours(tgt, layer);
                    if (tgtNeighbours.size() > M) {
                        NNVector candidates;
                        candidates.reserve(tgtNeighbours.size());
                        for (IdType nid : tgtNeighbours) {
                            candidates.emplace_back(nid, space->distance(value, nid));
                        }
                        IdVector newNeighbours =
                            selectNeighboursHeuristic(value, candidates, M, layer);
                        graph.setNeighbours(tgt, std::move(newNeighbours), layer);
                    }
                }
                graph.setNeighbours(id, std::move(neighbours), layer);

                if (layer == 0) {
                    break;
                }

                // reset enterPoints to candidates
                enterPoints.clear();
                std::transform(candidates.begin(), candidates.end(),
                               std::inserter(enterPoints, enterPoints.begin()),
                               [](const auto & p) { return p.first; });
                layer--;
            }

            if (maxLayer > graph.maxLayer) {
                graph.enterPoint = id;
                graph.maxLayer = maxLayer;
            }
        }

        bool remove(LabelType label) {
            IdType id;
            {
                std::unique_lock insertLock(insertMutex);
                id = space->getId(label);
                if (id == INVALID_ID) {
                    // label not found
                    return false;
                }

                graph.removeNode(id);
            }

            {
                std::unique_lock maxLayerLock(*graph.getMaxLayerMutex());
                if (graph.enterPoint == id) {
                    graph.refreshEnterPoint();
                }
            }

            for (size_t layer = 0; layer <= graph.getMaxLayer(id); layer++) {
                IdVector neighbours = graph.getNeighbours(id, layer);

                std::unordered_set<IdType> candidateSet;
                for (IdType nid : neighbours) {
                    candidateSet.insert(nid);
                    std::unique_lock nidLock(*graph.getMutex(nid));
                    for (IdType nnid : graph.getNeighbours(nid, layer)) {
                        candidateSet.insert(nnid);
                    }
                }

                NNVector candidates;
                candidates.reserve(config.efConstruction);
                for (IdType nid : neighbours) {
                    const ValueType & nvalue = space->getValue(nid);
                    for (IdType cid : candidateSet) {
                        if (cid == nid) {
                            continue;
                        }

                        dist_t dist = space->distance(nvalue, cid);
                        if (candidates.size() < config.efConstruction) {
                            candidates.emplace_back(cid, dist);
                            if (candidates.size() == config.efConstruction) {
                                std::make_heap(candidates.begin(), candidates.end(),
                                               maxHeapCompare);
                            }
                        } else if (dist < candidates.front().second) {
                            std::pop_heap(candidates.begin(), candidates.end(),
                                          maxHeapCompare);
                            candidates.back() = std::make_pair(cid, dist);
                            std::push_heap(candidates.begin(), candidates.end(),
                                           maxHeapCompare);
                        }
                    }

                    const size_t M = (layer > 0) ? config.M_ : config.M0_;
                    IdVector newNeighbours =
                        selectNeighboursHeuristic(nvalue, candidates, M, layer);

                    {
                        std::unique_lock nidLock(*graph.getMutex(nid));
                        if (!graph.isDeleted(nid)) {
                            graph.setNeighbours(nid, std::move(newNeighbours), layer, true);
                        }
                    }
                    // reset candidates
                    candidates.clear();
                }
            }

            {
                std::unique_lock insertLock(insertMutex);
                space->remove(id);
            }

            return true;
        }

        using NNQueryResult = std::vector<std::pair<LabelType, dist_t>>;

        NNQueryResult searchKNN(const ValueType & query, size_t k) const {
            IdType currEnterPoint = graph.enterPoint;
            if (currEnterPoint == INVALID_ID) {
                return NNQueryResult();
            }

            for (size_t layer = graph.maxLayer; layer > 0; layer--) {
                NNVector found = searchLayer(query, currEnterPoint, 1, layer);
                currEnterPoint = found.front().first;
            }
            NNVector found = searchLayer(query, currEnterPoint, std::max(k, config.ef));

            const size_t retSize = std::min(k, found.size());
            std::partial_sort(found.begin(), found.begin() + retSize, found.end(), maxHeapCompare);

            NNQueryResult ret;
            ret.reserve(retSize);
            for (size_t idx = 0; idx < retSize; idx++) {
                const auto & [id, dist] = found[idx];
                ret.emplace_back(space->getLabel(id), dist);
            }
            return ret;
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

            NNPair entry(enterPoint, space->distance(query, enterPoint));
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
                dist_t dist = space->distance(query, pid);
                candidates.emplace_back(pid, dist);
                found.emplace_back(pid, dist);
            }
            std::make_heap(candidates.begin(), candidates.end(), minHeapCompare);
            std::make_heap(found.begin(), found.end(), maxHeapCompare);

            return searchLayerImpl(query, candidates, found, visited, ef, layer);
        }


        NNVector searchLayerImpl(const ValueType & query, NNVector & candidates,
                                 NNVector & found, std::unordered_set<IdType> & visited,
                                 size_t ef, size_t layer = 0) const {
            // Assumes candidates (resp. found) is a min (resp. max) heap
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

                std::unique_lock cidLock(*graph.getMutex(cid));
                for (IdType nid : graph.getNeighbours(cid, layer)) {
                    auto it = visited.insert(nid);
                    if (!it.second) {
                        // already visited
                        continue;
                    }

                    dist_t ndist = space->distance(query, nid);
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
                                           size_t M, size_t layer, bool extendCandidates = false,
                                           bool keepPrunedConnections = false) const {
            NNVector candidatesMinHeap(candidates);
            if (extendCandidates) {
                std::unordered_set<IdType> visited;
                std::transform(candidates.begin(), candidates.end(),
                               std::inserter(visited, visited.begin()),
                               [](const auto & p) { return p.first; });

                for (const NNPair & c : candidates) {
                    std::unique_lock cidLock(*graph.getMutex(c.first));
                    for (IdType nid : graph.getNeighbours(c.first, layer)) {
                        if (auto it = visited.find(nid); it == visited.end()) {
                            visited.insert(nid);
                            dist_t ndist = space->distance(query, nid);
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
            if (keepPrunedConnections) {
                pruned.reserve(candidatesMinHeap.size());
            }
            while (!candidatesMinHeap.empty() && ret.size() < M) {
                auto [cid, cdist] = candidatesMinHeap.front();
                std::pop_heap(candidatesMinHeap.begin(), candidatesMinHeap.end(), minHeapCompare);
                candidatesMinHeap.pop_back();

                bool include = true;
                const ValueType & cvalue = space->getValue(cid);
                // heuristic: include only points that are closer to query
                // than to any other previously returned values
                for (IdType rid : ret) {
                    dist_t dist = space->distance(cvalue, rid);
                    if (dist < cdist) {
                        include = false;
                        break;
                    }
                }

                if (include) {
                    ret.push_back(cid);
                } else if (keepPrunedConnections) {
                    pruned.push_back(cid);
                }
            }

            if (keepPrunedConnections && ret.size() < M) {
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

     private:
        const HnswConfig & config;
        HNSWGraph graph;
        SpaceType * space;

        mutable std::mutex insertMutex;

        std::mt19937 layerGen;
    };
} // namespace hnsw
#endif // HNSW_H
