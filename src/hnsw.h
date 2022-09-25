#ifndef HNSW_H
#define HNSW_H

#include <functional>
#include <queue>
#include <unordered_set>
#include <vector>

#include "graph.h"
#include "space.h"

namespace hnsw {
    struct HnswConfig {
        size_t capacity = 128;

        size_t M_ = 16;
        size_t M0_ = 32;

        bool keepPrunedConnections = false;
    };

    template<typename dist_t, typename _ValueType>
    class HnswIndex {
     public:
        using SpaceType = MetricSpace<dist_t, _ValueType>;
        using ValueType = _ValueType;

        HnswIndex(const HnswConfig & config, const SpaceType & space)
            : config(config)
            , space(space)
            , size(0)
            , capacity(config.capacity)
            , graph(config.M_, config.M0_) {
            elements.reserve(capacity);
        }

        const HnswConfig & config;
        const SpaceType & space;

        using NNPair = std::pair<IdType, dist_t>;

        template<typename Compare = std::less<dist_t>>
        struct CompareNNPair {
            Compare cmp;

            bool operator() (const NNPair & a, const NNPair & b) const {
                return cmp(a.second, b.second);
            }
        };

        template<typename Compare = std::less<dist_t>>
        using NNQueue = std::priority_queue<NNPair, std::vector<NNPair>, CompareNNPair<Compare>>;

        using NNMaxHeap = NNQueue<>;
        using NNMinHeap = NNQueue<std::greater<dist_t>>;

        NNMaxHeap searchLayer(const ValueType & query,
                              const std::unordered_set<IdType> & enterPoints,
                              size_t ef, size_t layer) const {
            std::unordered_set<IdType> visited = enterPoints;

            NNMinHeap candidates;
            NNMaxHeap found;
            for (IdType pid : enterPoints) {
                dist_t dist = distance(query, pid);
                candidates.emplace(pid, dist);
                found.emplace(pid, dist);
            }

            while (!candidates.empty()) {
                // closest candidate to query
                auto [cid, cdist] = candidates.top();
                candidates.pop();
                // farthest found point to query
                auto fdist = found.top().second;
                if (cdist > fdist) {
                    // we are done here
                    break;
                }

                for (IdType nid : graph.neighbours(cid, layer)) {
                    auto it = visited.insert(nid);
                    if (!it.second) {
                        // already visited
                        continue;
                    }

                    dist_t ndist = distance(query, nid);
                    if (ndist < fdist or found.size() < ef) {
                        candidates.emplace(nid, ndist);
                        found.emplace(nid, ndist);
                        if (found.size() > ef) {
                            found.pop();
                        }
                    }

                    // update farthest point
                    fdist = found.top().second;
                }
            }

            return found;
        }

        std::vector<IdType> selectNeighboursHeuristic(NNMaxHeap * maxHeap, size_t M) const {
            std::vector<IdType> ret;
            if (maxHeap->size() < M) {
                ret.reserve(maxHeap->size());
                while (!maxHeap->empty()) {
                    ret.push_back(maxHeap->top().first);
                    maxHeap->pop();
                }
                return ret;
            }

            // transfer max heap to min heap
            NNMinHeap candidates;
            while (!maxHeap->empty()) {
                candidates.push(maxHeap->top());
                maxHeap->pop();
            }

            ret.reserve(M);
            std::vector<IdType> pruned;
            while (!candidates.empty() && ret.size() < M) {
                auto [cid, cdist] = candidates.top();
                candidates.pop();

                bool include = true;
                const ValueType & cvalue = (*this)[cid];
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
        std::vector<ValueType> elements;
        size_t size;
        size_t capacity;

        HNSWGraph graph;
    };
} // namespace  hnsw
#endif // HNSW_H