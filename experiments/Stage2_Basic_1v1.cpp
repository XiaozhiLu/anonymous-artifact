#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <set>
#include <queue>
#include <chrono>
#include <iomanip>
#include <cmath>


#include "retrieval_framework.hpp"

using namespace std;
using namespace anns::graph;


float* load_fvecs(const string& filename, size_t& num, size_t& dim) {
    ifstream in(filename, ios::binary);
    if (!in.is_open()) { cerr << "Error: Cannot open " << filename << endl; exit(1); }
    int dim_int; in.read((char*)&dim_int, 4); dim = (size_t)dim_int;
    in.seekg(0, ios::end); size_t file_size = in.tellg(); in.seekg(0, ios::beg);
    size_t row_size = 4 + 4 * dim; num = file_size / row_size;
    float* data = new float[num * dim];
    in.seekg(0, ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
    cout << "Loaded " << filename << ": " << num << " vectors, dim=" << dim << endl;
    return data;
}


vector<vector<int>> ComputeGroundTruth(const float* base, const float* query, size_t n_base, size_t n_query, size_t dim, int K) {
    cout << "Computing Ground Truth in C++ (Linear Scan)..." << endl;
    vector<vector<int>> gt(n_query);
    #pragma omp parallel for
    for (int i = 0; i < (int)n_query; ++i) {
        const float* q_vec = query + i * dim;
        std::priority_queue<std::pair<float, int>> pq;
        for (size_t j = 0; j < n_base; ++j) {
            const float* b_vec = base + j * dim;
            float dist = vec_L2sqr(q_vec, b_vec, dim);
            if (pq.size() < K) { pq.push({dist, (int)j}); }
            else if (dist < pq.top().first) { pq.pop(); pq.push({dist, (int)j}); }
        }
        gt[i].resize(K);
        for (int k = K - 1; k >= 0; --k) { gt[i][k] = pq.top().second; pq.pop(); }
    }
    cout << "Ground Truth Computed." << endl;
    return gt;
}


void GenerateSocialGraph(SocialGraph& sg, size_t num_nodes, int edges_per_node) {
    cout << "Generating Synthetic Social Graph (Scale-Free)..." << endl;
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, num_nodes - 1);
    int num_influencers = num_nodes / 100; // 1% 大V
    if (num_influencers == 0) num_influencers = 1;
    for (int i = 0; i < num_nodes; ++i) {
        for (int k = 0; k < edges_per_node; ++k) {
            int target = (k == 0) ? (rng() % num_influencers) : dist(rng);
            if (i != target) sg.add_edge(i, target);
        }
    }
    cout << "Social Graph Generated." << endl;
}


float CalculateRecall(const vector<int>& results, const vector<int>& ground_truth, int K) {
    if (results.empty()) return 0.0f;
    std::set<int> gt_set(ground_truth.begin(), ground_truth.end());
    int hits = 0;
    for (int id : results) if (gt_set.count(id)) hits++;
    return (float)hits / K;
}

float CalculateInfluence(const vector<int>& results, const SocialGraph& sg) {
    std::set<int> union_neighbors;
    for (int id : results) {
        for (int nb : sg.get_neighbors(id)) {
            union_neighbors.insert(nb);
        }
    }
    return (float)union_neighbors.size();
}

int main() {

    string base_path = "data/sift-128-euclidean.train.fvecs";
    string query_path = "data/sift-128-euclidean.test.fvecs";

    size_t dim_base, num_base, dim_query, num_query;
    float* data_base = load_fvecs(base_path, num_base, dim_base);
    float* data_query = load_fvecs(query_path, num_query, dim_query);

    int num_test = 100;
    int K = 10;
    auto ground_truth = ComputeGroundTruth(data_base, data_query, num_base, num_test, dim_base, K);

    SocialGraph sg(num_base);
    GenerateSocialGraph(sg, num_base, 10);


    cout << "Building Standard HNSW Index..." << endl;
    HNSW<float> index(dim_base, num_base, 16, 200);
    index.SetSocialGraph(&sg);
    index.SetAlphaParams(0.0f, false, 200000.0f, 300.0f);

    #pragma omp parallel for
    for (size_t i = 0; i < num_base; ++i) {
        index.AddPoint(data_base + i * dim_base);
    }
    index.SetReady(true);
    cout << "Index built." << endl;



    cout << "\n>>> Starting Experiment: Pure Scheme 1v1 <<<" << endl;

    float epsilon = 0.2f;
    std::vector<float> beta_list = {0.2f, 0.5f, 0.8f, 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 50.0f, 100.0f};

    cout << "---------------------------------------------------------------" << endl;
    cout << std::left << std::setw(10) << "Beta"
         << std::setw(15) << "Recall@10"
         << std::setw(15) << "Avg_Influence"
         << std::setw(15) << "Time(ms)" << endl;
    cout << "---------------------------------------------------------------" << endl;

    for (float b : beta_list) {
        float total_recall = 0;
        float total_inf = 0;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_test; ++i) {

            std::vector<int> res = index.SearchIterativeOriginal1v1(data_query + i * dim_query, K, epsilon, b);

            total_recall += CalculateRecall(res, ground_truth[i], K);
            total_inf += CalculateInfluence(res, sg);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_test;

        cout << std::left << std::setw(10) << b
             << std::setw(15) << (total_recall / num_test)
             << std::setw(15) << (total_inf / num_test)
             << std::setw(15) << time_ms << endl;
    }

    cout << "---------------------------------------------------------------" << endl;

    delete[] data_base;
    delete[] data_query;
    return 0;
}
