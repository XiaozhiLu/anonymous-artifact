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
#include <cstdint>

#include "retrieval_framework.hpp"

using namespace std;
using namespace anns::graph;


float* load_embedding_bin(const string& filename, size_t& num, size_t& dim) {
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }

    int32_t n = 0, d = 0;
    in.read((char*)&n, sizeof(int32_t));
    in.read((char*)&d, sizeof(int32_t));

    if (!in || n <= 0 || d <= 0) {
        cerr << "Error: Invalid embedding file header: " << filename << endl;
        exit(1);
    }

    num = (size_t)n;
    dim = (size_t)d;

    float* data = new float[num * dim];
    in.read((char*)data, sizeof(float) * num * dim);

    if (!in) {
        cerr << "Error: Failed to read embedding matrix from " << filename << endl;
        delete[] data;
        exit(1);
    }

    in.close();
    cout << "Loaded " << filename << ": " << num << " vectors, dim=" << dim << endl;
    return data;
}


vector<int> LoadQueryItems(const string& filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }

    vector<int> query_items;
    int item_id;
    while (in >> item_id) {
        query_items.push_back(item_id);
    }
    in.close();

    cout << "Loaded " << filename << ": " << query_items.size() << " query items." << endl;
    return query_items;
}


void LoadRealSocialGraph(SocialGraph& sg, const string& filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }

    int u, v;
    size_t edge_cnt = 0;
    while (in >> u >> v) {
        sg.add_edge(u, v);
        edge_cnt++;
    }
    in.close();

    cout << "Loaded real social graph from " << filename
         << ", edges=" << edge_cnt << endl;
}


vector<vector<int>> ComputeGroundTruth(
    const float* base,
    const float* query,
    const vector<int>& query_ids,
    size_t n_base,
    size_t dim,
    int K
) {
    cout << "Computing Ground Truth in C++ (Linear Scan)..." << endl;

    size_t n_query = query_ids.size();
    vector<vector<int>> gt(n_query);

    #pragma omp parallel for
    for (int i = 0; i < (int)n_query; ++i) {
        int qid = query_ids[i];
        const float* q_vec = query + (size_t)qid * dim;

        std::priority_queue<std::pair<float, int>> pq;

        for (size_t j = 0; j < n_base; ++j) {
            const float* b_vec = base + j * dim;
            float dist = vec_L2sqr(q_vec, b_vec, dim);

            if ((int)pq.size() < K) {
                pq.push({dist, (int)j});
            } else if (dist < pq.top().first) {
                pq.pop();
                pq.push({dist, (int)j});
            }
        }

        gt[i].resize(K);
        for (int k = K - 1; k >= 0; --k) {
            gt[i][k] = pq.top().second;
            pq.pop();
        }
    }

    cout << "Ground Truth Computed." << endl;
    return gt;
}


float CalculateRecall(const vector<int>& results, const vector<int>& ground_truth, int K) {
    if (results.empty()) return 0.0f;
    std::set<int> gt_set(ground_truth.begin(), ground_truth.end());
    int hits = 0;
    for (int id : results) {
        if (gt_set.count(id)) hits++;
    }
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

    string base_path = "processed_ciao/user_emb.bin";
    string query_path = "processed_ciao/item_emb.bin";
    string trust_path = "processed_ciao/trust_edges.txt";
    string query_items_path = "processed_ciao/query_items.txt";

    size_t dim_base, num_base, dim_query, num_query;
    float* data_base = load_embedding_bin(base_path, num_base, dim_base);   // user embeddings
    float* data_query = load_embedding_bin(query_path, num_query, dim_query); // item embeddings

    if (dim_base != dim_query) {
        cerr << "Dimension Mismatch!" << endl;
        return -1;
    }


    vector<int> all_query_items = LoadQueryItems(query_items_path);

    int num_test = 100;
    if ((int)all_query_items.size() < num_test) {
        num_test = (int)all_query_items.size();
    }

    vector<int> test_query_items(all_query_items.begin(), all_query_items.begin() + num_test);

    int K = 10;
    auto ground_truth = ComputeGroundTruth(data_base, data_query, test_query_items, num_base, dim_base, K);

    SocialGraph sg(num_base);
    LoadRealSocialGraph(sg, trust_path);


    cout << "Building Standard HNSW Index (Pure Physical)..." << endl;
    size_t M = 16;
    size_t ef_construct = 200;

    HNSW<float> index(dim_base, num_base, M, ef_construct);
    index.SetSocialGraph(&sg);


    index.SetAlphaParams(0.0f, false, 200000.0f, 300.0f);

    #pragma omp parallel for
    for (size_t i = 0; i < num_base; ++i) {
        index.AddPoint(data_base + i * dim_base);
    }
    index.SetReady(true);
    cout << "Index built successfully." << endl;


    cout << "\n=================================================================" << endl;
    cout << "   Self-Decision Adaptive Strategies Evaluation (CiaoDVD Dataset)" << endl;
    cout << "=================================================================" << endl;

    std::vector<float> target_recalls = {0.70f, 0.80f, 0.90f};
    float epsilon = 0.2f;


    cout << "\n[Strategy : Proxy-Metric Feedback Loop]" << endl;
    cout << std::left << std::setw(15) << "Target_Recall"
         << std::setw(15) << "Actual_Recall"
         << std::setw(15) << "Avg_Influence"
         << std::setw(15) << "Time(ms)" << endl;
    cout << "-----------------------------------------------------------------" << endl;

    for (float t : target_recalls) {
        float total_recall = 0;
        float total_inf = 0;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_test; ++i) {
            int qid = test_query_items[i];
            const float* q_ptr = data_query + (size_t)qid * dim_query;

            auto res = index.SearchIterative_Strategy_final(q_ptr, K, epsilon, t);
            total_recall += CalculateRecall(res, ground_truth[i], K);
            total_inf += CalculateInfluence(res, sg);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_test;

        cout << std::left << std::setw(15) << t
             << std::setw(15) << (total_recall / num_test)
             << std::setw(15) << (total_inf / num_test)
             << std::setw(15) << time_ms << endl;
    }

    cout << "=================================================================" << endl;

    delete[] data_base;
    delete[] data_query;
    return 0;
}
