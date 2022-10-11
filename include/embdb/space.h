#ifndef SPACE_H
#define SPACE_H

#include <queue>
#include <unordered_set>
#include <vector>

#include "common.h"

namespace hnsw {
    template<typename dist_t, typename _ValueType, typename LabelType = size_t>
    class MetricSpace {
     public:
        MetricSpace()
            : capacity_(0)
            , size_(0) {}

        virtual ~MetricSpace() = default;

        using ValueType = _ValueType;

        // not necessarily a true distance but can be a monotonically increasing function of it,
        // e.g., can be L2 distance squared
        virtual dist_t distance(const ValueType & obj1, const ValueType & obj2) const = 0;

        dist_t distance(const ValueType & query, IdType id) const {
            return distance(query, elements[id]);
        }

        dist_t distance(IdType id1, IdType id2) const {
            return distance(elements[id1], elements[id2]);
        }

        void setCapacity(size_t capacity) {
            labels.reserve(capacity);
            elements.reserve(capacity);
            capacity_ = capacity;
        }

        size_t capacity() const {
            return capacity_;
        }

        size_t size() const {
            return size_;
        }

        ValueType getValue(IdType id) const {
            return elements[id];
        }

        LabelType getLabel(IdType id) const {
            return labels[id];
        }

        IdType getId(LabelType label) const {
            auto it = labelsInv.find(label);
            return (it == labelsInv.end()) ? INVALID_ID : it->second;
        }

        virtual IdType add(const ValueType & value, LabelType label) {
            if (labelsInv.find(label) != labelsInv.end()) {
                // label already present
                return INVALID_ID;
            }

            IdType id;
            if (!deleted.empty()) {
                id = deleted.front();
                deleted.pop();
                labels[id] = label;
                elements[id] = value;
            } else {
                if (size_ >= capacity_) {
                    throw std::runtime_error("Capacity reached -- index must be resized");
                }
                id = size_;
                labels.push_back(label);
                elements.push_back(value);
            }

            labelsInv[label] = id;
            size_++;

            return id;
        }

        virtual void remove(IdType id) {
            labelsInv.erase(labels[id]);
            deleted.push(id);
            size_--;
        }

        virtual std::unordered_set<IdType> checkIntegrity() const {
            std::unordered_set<IdType> ret;
            std::queue<IdType> deletedCopy(deleted);
            while (!deletedCopy.empty()) {
                IdType did = deletedCopy.front();
                deletedCopy.pop();
                ret.insert(did);
            }

            size_t _deleted = 0, valid = 0;
            for (IdType id = 0; id < labels.size(); id++) {
                if (auto it = ret.find(id); it != ret.end()) {
                    _deleted++;
                    continue;
                }

                valid++;
                LabelType label = labels[id];
                auto it = labelsInv.find(label);
                if (it == labelsInv.end() || it->second != id) {
                    throw std::runtime_error("Invalid id-label pair: " + std::to_string(id) +
                                             "-" + std::to_string(label));
                }
            }

            if (valid != size_ || valid != labelsInv.size() || _deleted != deleted.size()) {
                throw std::runtime_error("Size discrepancy");
            }

            return ret;
        }

     protected:
        std::vector<LabelType> labels;
        std::unordered_map<LabelType, IdType> labelsInv;
        std::vector<ValueType> elements;

        std::queue<IdType> deleted;

        size_t capacity_;
        size_t size_;
    };

    template<typename dist_t = float, typename entry_t = dist_t, typename LabelType = size_t>
    class EmbeddingMetricSpace : public MetricSpace<dist_t, std::vector<entry_t>, LabelType> {
     public:
        explicit EmbeddingMetricSpace(size_t dim)
            : dim(dim) {}

        using Base = MetricSpace<dist_t, std::vector<entry_t>, LabelType>;
        using ValueType = typename Base::ValueType;

        IdType add(const ValueType & value, LabelType label) override {
            if (value.size() != dim) {
                throw std::runtime_error("Dimension mismatch");
            }
            return Base::add(value, label);
        }

        const size_t dim;
    };

    template<typename dist_t = float, typename entry_t = dist_t, typename LabelType = size_t>
    class L2Space : public EmbeddingMetricSpace<dist_t, entry_t, LabelType> {
     public:
        explicit L2Space(size_t dim)
            : EmbeddingMetricSpace<dist_t, entry_t, LabelType>(dim) {}

        using Base = EmbeddingMetricSpace<dist_t, entry_t, LabelType>;
        using ValueType = typename Base::ValueType;

        dist_t distance(const ValueType & obj1, const ValueType & obj2) const override {
            dist_t ret = 0;
            dist_t diff;
            for (size_t i = 0; i < this->dim; i++) {
                diff = obj1[i] - obj2[i];
                ret += diff * diff;
            }
            return ret;
        }
    };
} // namespace hnsw
#endif // SPACE_H