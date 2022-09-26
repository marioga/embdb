#include <iostream>
#include <fstream>

#include "hnsw.h"

void openBinaryFile(const std::string & filename, std::ifstream * istrm) {
    istrm->open(filename, std::ios::binary);
    if (!istrm->is_open()) {
        std::cout << "failed to open " << filename << std::endl;
    }
}

template<typename T>
bool readVector(std::ifstream * istrm, int expectedDim, std::vector<T> & vec) {
    int currDim;
    if (!istrm->read(reinterpret_cast<char *>(&currDim), sizeof(currDim))) {
        return false;
    }

    if (currDim != expectedDim) {
        throw std::runtime_error("Invalid base dimension");
    }

    istrm->read(reinterpret_cast<char *>(vec.data()), currDim * sizeof(T));
    return true;
}

void siftTest() {
    const int dim = 128;
    const int k = 16;
    using _L2Space = hnsw::L2Space<float>;
    _L2Space space = _L2Space(dim);

    using ValueType = _L2Space::ValueType;
    using IndexType = hnsw::HnswIndex<float, ValueType>;

    hnsw::HnswConfig config;
    // config.capacity = 10000;
    config.capacity = 1000000;
    config.ef = 128;

    std::cout << "Building index..." << std::endl;
    IndexType index = IndexType(config, space);
    {
        std::ifstream istrm_base;
        // openBinaryFile("siftsmall/siftsmall_base.fvecs", &istrm_base);
        openBinaryFile("sift/sift_base.fvecs", &istrm_base);

        size_t label = 0;
        ValueType value(dim);
        while (readVector(&istrm_base, dim, value)) {
            index.insert(value, label++);
            if (label % 50000 == 0) {
                std::cout << "Added " << label << " vectors" << std::endl;
            }
        }
    }
    std::cout << "Completed index build" << std::endl;

    std::ifstream istrm_gt;
    std::ifstream istrm_query;
    // openBinaryFile("siftsmall/siftsmall_groundtruth.ivecs", &istrm_gt);
    // openBinaryFile("siftsmall/siftsmall_query.fvecs", &istrm_query);
    openBinaryFile("sift/sift_groundtruth.ivecs", &istrm_gt);
    openBinaryFile("sift/sift_query.fvecs", &istrm_query);

    const size_t gtDim = 100;
    std::vector<int> gt(gtDim);
    ValueType query(dim);
    std::unordered_set<hnsw::IdType> gtKNN;
    size_t correct = 0, total = 0;
    while (readVector(&istrm_gt, gtDim, gt) && readVector(&istrm_query, dim, query)) {
        gtKNN.clear();
        for (size_t i = 0; i < k; i++) {
            gtKNN.insert(gt[i]);
        }

        IndexType::NNVector ret = index.searchKNN(query, 16);
        for (const auto & entry : ret) {
            if (gtKNN.find(entry.first) != gtKNN.end()) {
                correct++;
            }
        }
        total += k;
    }

    std::cout << "Recall: " << (100.0f * correct) / total << "%" << std::endl;
}

int main() {
    siftTest();
}