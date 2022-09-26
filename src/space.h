#ifndef SPACE_H
#define SPACE_H

#include <vector>

namespace hnsw {
    template<typename dist_t, typename _ValueType>
    class MetricSpace {
     public:
        virtual ~MetricSpace() = default;

        using ValueType = _ValueType;

        // not necessarily a true distance but can be a monotonically increasing function of it,
        // e.g., can be L2 distance squared
        virtual dist_t distance(const ValueType & obj1, const ValueType & obj2) const = 0;
    };

    template<typename dist_t = float, typename entry_t = dist_t>
    class EmbeddingMetricSpace : public MetricSpace<dist_t, std::vector<entry_t>> {
     public:
        explicit EmbeddingMetricSpace(size_t dim)
            : dim(dim) {}

        using Base = MetricSpace<dist_t, std::vector<entry_t>>;
        using ValueType = typename Base::ValueType;

        size_t dim;
    };

    template<typename dist_t = float, typename entry_t = dist_t>
    class L2Space : public EmbeddingMetricSpace<dist_t, entry_t> {
     public:
        explicit L2Space(size_t dim)
            : EmbeddingMetricSpace<dist_t, entry_t>(dim) {}

        using Base = EmbeddingMetricSpace<dist_t, entry_t>;
        using ValueType = typename Base::ValueType;

        dist_t distance(const ValueType & obj1, const ValueType & obj2) const override {
            dist_t ret = 0;
            for (size_t i = 0; i < this->dim; i++) {
                ret += (obj1[i] - obj2[i]) * (obj1[i] - obj2[i]);
            }
            return ret;
        }
    };
} // namespace  hnsw
#endif // SPACE_H