#ifndef SPACE_H
#define SPACE_H

#include <vector>

#include "common.h"

namespace hnsw {
    template<typename dist_t, typename _ValueType, typename LabelType = size_t>
    class MetricSpace {
     public:
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
            this->capacity = capacity;
        }

        size_t size() const {
            return labels.size();
        }

        ValueType getValue(IdType id) const {
            return elements[id];
        }

        LabelType getLabel(IdType id) const {
            return labels[id];
        }

        IdType getId(LabelType label) const {
            auto it = labelsInv.find(label);
            return (it != labelsInv.end()) ? it->second : INVALID_ID;
        }

        virtual IdType add(const ValueType & value, LabelType label) {
            IdType id = size();
            if (id >= capacity) {
                throw std::runtime_error("Capacity reached -- index must be resized");
            }

            elements.push_back(value);
            labels.push_back(label);
            labelsInv[label] = id;

            return id;
        }

     protected:
        std::vector<LabelType> labels;
        std::unordered_map<LabelType, IdType> labelsInv;
        std::vector<ValueType> elements;

        size_t capacity;
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