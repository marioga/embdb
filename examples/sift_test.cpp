#include <iostream>
#include <fstream>

#include "embdb/hnsw.h"

void openBinaryFile(const std::string & filename, std::ifstream * istrm) {
    istrm->open(filename, std::ios::binary);
    if (!istrm->is_open()) {
        throw std::runtime_error("Failed to open " + filename);
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

    std::ifstream istrm_base;
    // openBinaryFile("siftsmall/siftsmall_base.fvecs", &istrm_base);
    openBinaryFile("sift/sift_base.fvecs", &istrm_base);

    ValueType value(dim);
    std::vector<ValueType> allValues;
    allValues.reserve(config.capacity);
    while (readVector(&istrm_base, dim, value)) {
        allValues.push_back(value);
    }

    std::cout << "Building index..." << std::endl;
    IndexType index = IndexType(config, &space);
    hnsw::StopWatch sw;
    size_t count = 0;
    #pragma omp parallel for
    for (size_t idx = 0; idx < allValues.size(); idx++) {
        index.insert(allValues[idx], idx);
        #pragma omp atomic
        ++count;
        if (count % 50000 == 0) {
            std::cout << "Indexed " << count << " vectors" << std::endl;
        }
    }
    std::cout << "Completed index build -- size: " << index.size() << " -- time elapsed: "
        << sw.elapsed<std::chrono::seconds>() << "s" << std::endl;

    std::ifstream istrm_gt;
    // openBinaryFile("siftsmall/siftsmall_groundtruth.ivecs", &istrm_gt);
    openBinaryFile("sift/sift_groundtruth.ivecs", &istrm_gt);
    const size_t gtDim = 100;
    std::vector<int> gt(gtDim);
    std::vector<std::unordered_set<int>> allGTKNN;
    while (readVector(&istrm_gt, gtDim, gt)) {
        allGTKNN.emplace_back(gt.begin(), gt.begin() + k);
    }

    std::ifstream istrm_query;
    // openBinaryFile("siftsmall/siftsmall_query.fvecs", &istrm_query);
    openBinaryFile("sift/sift_query.fvecs", &istrm_query);

    ValueType query(dim);
    std::vector<ValueType> allQueries;
    while (readVector(&istrm_query, dim, query)) {
        allQueries.push_back(query);
    }

    std::vector<std::vector<size_t>> rets;
    rets.resize(allQueries.size());
    sw.reset();
    #pragma omp parallel for
    for (size_t idx = 0; idx < allQueries.size(); idx++) {
        auto ret = index.searchKNN(allQueries[idx], k);
        for (const auto & entry : ret) {
            rets[idx].push_back(entry.first);
        }
    }
    std::cout << "Querying " << allQueries.size() << " items -- time elapsed: "
        << sw.elapsed<std::chrono::milliseconds>() << "ms" << std::endl;

    size_t correct = 0, total = 0;
    for (size_t idx = 0; idx < allQueries.size(); idx++) {
        const auto & ret = rets[idx];
        const auto & gtKNN = allGTKNN[idx];
        for (auto label : ret) {
            if (gtKNN.find(label) != gtKNN.end()) {
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