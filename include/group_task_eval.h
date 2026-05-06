#ifndef GROUP_TASK_EVAL_H
#define GROUP_TASK_EVAL_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct GroupTaskEvalResult {
    int query_id = -1;
    std::string method;

    double constraint_recall = 0.0;   // 你已有的 recall 约束完成度
    int overlap_hits = 0;             // 与 proxy truth 的重合个数
    int min_overlap_required = 0;
    int positives_hit = 0;            // 命中的真实响应用户数

    double precision_at_k = 0.0;
    double recall_at_k = 0.0;
    double hitrate_at_k = 0.0;
    double ndcg_at_k = 0.0;

    int spread_at_k = 0;              // 你选出的群体的一跳去重覆盖
    int ideal_spread_at_k = 0;        // 在相同候选池 + 相同 recall 约束下的近似最优覆盖
    double nspread_at_k = 0.0;        // = spread / ideal_spread

    int pool_spread_upper = 0;        // 整个候选池的一跳并集，仅作参考
    double pool_cover_rate = 0.0;     // = spread / pool_spread_upper
};

inline std::vector<int> UniqueKeepOrder(const std::vector<int>& ids) {
    std::vector<int> out;
    out.reserve(ids.size());
    std::unordered_set<int> seen;
    for (int x : ids) {
        if (seen.insert(x).second) out.push_back(x);
    }
    return out;
}

inline std::unordered_set<int> ToSet(const std::vector<int>& ids) {
    return std::unordered_set<int>(ids.begin(), ids.end());
}

inline std::vector<std::string> SplitCSVLine(const std::string& line) {
    std::vector<std::string> cols;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '\"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            cols.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    cols.push_back(cur);
    return cols;
}

inline std::unordered_map<int, std::unordered_set<int>> LoadQueryUserSetCSV(
    const std::string& path,
    int query_col = 0,
    int user_col = 1,
    int label_col = 2,
    int positive_label = 1,
    bool has_header = true) {

    std::unordered_map<int, std::unordered_set<int>> data;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[LoadQueryUserSetCSV] Cannot open file: " << path << std::endl;
        return data;
    }

    std::string line;
    if (has_header) std::getline(fin, line);

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cols = SplitCSVLine(line);
        if ((int)cols.size() <= std::max({query_col, user_col, label_col})) continue;

        int q = std::stoi(cols[query_col]);
        int u = std::stoi(cols[user_col]);
        int y = std::stoi(cols[label_col]);
        if (y == positive_label) data[q].insert(u);
    }
    return data;
}

template <typename GraphT>
inline std::unordered_set<int> BuildSpreadSet(const std::vector<int>& seeds,
                                              const GraphT* social_graph,
                                              bool include_seed_itself = true) {
    std::unordered_set<int> covered;
    if (!social_graph) return covered;

    for (int u : seeds) {
        if (include_seed_itself) covered.insert(u);
        const auto& nbs = social_graph->get_neighbors(u);
        for (int v : nbs) covered.insert(v);
    }
    return covered;
}

template <typename GraphT>
inline int SpreadAtK(const std::vector<int>& seeds,
                     const GraphT* social_graph,
                     bool include_seed_itself = true) {
    return (int)BuildSpreadSet(seeds, social_graph, include_seed_itself).size();
}

inline double PrecisionAtK(const std::vector<int>& ranked_ids,
                           const std::unordered_set<int>& positives,
                           int* hit_count = nullptr) {
    if (ranked_ids.empty()) {
        if (hit_count) *hit_count = 0;
        return 0.0;
    }
    int hits = 0;
    for (int u : ranked_ids) {
        if (positives.find(u) != positives.end()) ++hits;
    }
    if (hit_count) *hit_count = hits;
    return (double)hits / (double)ranked_ids.size();
}

inline double RecallAtK(const std::vector<int>& ranked_ids,
                        const std::unordered_set<int>& positives,
                        int* hit_count = nullptr) {
    if (positives.empty()) {
        if (hit_count) *hit_count = 0;
        return 0.0;
    }
    int hits = 0;
    for (int u : ranked_ids) {
        if (positives.find(u) != positives.end()) ++hits;
    }
    if (hit_count) *hit_count = hits;
    return (double)hits / (double)positives.size();
}

inline double HitRateAtK(const std::vector<int>& ranked_ids,
                         const std::unordered_set<int>& positives) {
    for (int u : ranked_ids) {
        if (positives.find(u) != positives.end()) return 1.0;
    }
    return 0.0;
}

inline double NDCGAtK(const std::vector<int>& ranked_ids,
                      const std::unordered_set<int>& positives) {
    if (ranked_ids.empty() || positives.empty()) return 0.0;

    double dcg = 0.0;
    for (size_t i = 0; i < ranked_ids.size(); ++i) {
        if (positives.find(ranked_ids[i]) != positives.end()) {
            dcg += 1.0 / std::log2((double)i + 2.0);
        }
    }

    const int ideal_hits = std::min((int)ranked_ids.size(), (int)positives.size());
    if (ideal_hits <= 0) return 0.0;

    double idcg = 0.0;
    for (int i = 0; i < ideal_hits; ++i) {
        idcg += 1.0 / std::log2((double)i + 2.0);
    }
    return (idcg <= 1e-12) ? 0.0 : (dcg / idcg);
}

inline int OverlapHits(const std::vector<int>& selected_ids,
                       const std::unordered_set<int>& proxy_truth) {
    int cnt = 0;
    for (int u : selected_ids) {
        if (proxy_truth.find(u) != proxy_truth.end()) ++cnt;
    }
    return cnt;
}

template <typename GraphT>
inline int MarginalSpreadGain(int node,
                              const std::unordered_set<int>& covered,
                              const GraphT* social_graph,
                              bool include_seed_itself = true) {
    if (!social_graph) return 0;
    int gain = 0;
    if (include_seed_itself && covered.find(node) == covered.end()) ++gain;
    const auto& nbs = social_graph->get_neighbors(node);
    for (int v : nbs) {
        if (covered.find(v) == covered.end()) ++gain;
    }
    return gain;
}

template <typename GraphT>
inline void AddNodeSpreadToCovered(int node,
                                   std::unordered_set<int>& covered,
                                   const GraphT* social_graph,
                                   bool include_seed_itself = true) {
    if (!social_graph) return;
    if (include_seed_itself) covered.insert(node);
    const auto& nbs = social_graph->get_neighbors(node);
    for (int v : nbs) covered.insert(v);
}

// 在“相同候选池 + 相同 recall floor”下，近似求一个可行的 ideal group。
// 1) 先从 proxy truth 交集里贪心选 min_overlap_required 个，确保 recall 约束；
// 2) 再从剩余候选里贪心补满 K 个，最大化 spread。
template <typename GraphT>
inline std::vector<int> GreedyFeasibleIdealGroup(const std::vector<int>& candidate_pool_ids,
                                                 const std::unordered_set<int>& proxy_truth,
                                                 size_t K,
                                                 int min_overlap_required,
                                                 const GraphT* social_graph,
                                                 bool include_seed_itself = true) {
    std::vector<int> pool = UniqueKeepOrder(candidate_pool_ids);
    std::vector<int> anchor_pool;
    std::vector<int> free_pool;
    anchor_pool.reserve(pool.size());
    free_pool.reserve(pool.size());

    for (int u : pool) {
        if (proxy_truth.find(u) != proxy_truth.end()) anchor_pool.push_back(u);
        else free_pool.push_back(u);
    }

    int feasible_overlap = std::min<int>(min_overlap_required, (int)anchor_pool.size());

    std::vector<int> selected;
    selected.reserve(K);
    std::unordered_set<int> covered;
    std::unordered_set<int> used;

    // Phase 1: 先满足 recall floor
    for (int step = 0; step < feasible_overlap; ++step) {
        int best_u = -1;
        int best_gain = -1;
        for (int u : anchor_pool) {
            if (used.find(u) != used.end()) continue;
            int gain = MarginalSpreadGain(u, covered, social_graph, include_seed_itself);
            if (gain > best_gain) {
                best_gain = gain;
                best_u = u;
            }
        }
        if (best_u == -1) break;
        selected.push_back(best_u);
        used.insert(best_u);
        AddNodeSpreadToCovered(best_u, covered, social_graph, include_seed_itself);
    }

    // Phase 2: 用所有剩余候选继续贪心补满 K
    while (selected.size() < K) {
        int best_u = -1;
        int best_gain = -1;
        for (int u : pool) {
            if (used.find(u) != used.end()) continue;
            int gain = MarginalSpreadGain(u, covered, social_graph, include_seed_itself);
            if (gain > best_gain) {
                best_gain = gain;
                best_u = u;
            }
        }
        if (best_u == -1) break;
        selected.push_back(best_u);
        used.insert(best_u);
        AddNodeSpreadToCovered(best_u, covered, social_graph, include_seed_itself);
    }

    return selected;
}

template <typename GraphT>
inline GroupTaskEvalResult EvaluateOneQueryGroup(
        int query_id,
        const std::string& method,
        const std::vector<int>& selected_ids_in,
        const std::vector<int>& candidate_pool_ids_in,
        const std::unordered_set<int>& proxy_truth,
        double target_recall_floor,
        const std::unordered_set<int>& positive_responders,
        const GraphT* social_graph,
        bool include_seed_itself = true) {

    GroupTaskEvalResult ret;
    ret.query_id = query_id;
    ret.method = method;

    const std::vector<int> selected_ids = UniqueKeepOrder(selected_ids_in);
    const std::vector<int> candidate_pool_ids = UniqueKeepOrder(candidate_pool_ids_in);

    const int K = (int)selected_ids.size();
    ret.min_overlap_required = (int)std::llround((double)K * target_recall_floor);
    ret.overlap_hits = OverlapHits(selected_ids, proxy_truth);
    ret.constraint_recall = (K > 0) ? ((double)ret.overlap_hits / (double)K) : 0.0;

    ret.precision_at_k = PrecisionAtK(selected_ids, positive_responders, &ret.positives_hit);
    ret.recall_at_k = RecallAtK(selected_ids, positive_responders, nullptr);
    ret.hitrate_at_k = HitRateAtK(selected_ids, positive_responders);
    ret.ndcg_at_k = NDCGAtK(selected_ids, positive_responders);

    ret.spread_at_k = SpreadAtK(selected_ids, social_graph, include_seed_itself);
    ret.pool_spread_upper = SpreadAtK(candidate_pool_ids, social_graph, include_seed_itself);
    ret.pool_cover_rate = (ret.pool_spread_upper > 0)
            ? (double)ret.spread_at_k / (double)ret.pool_spread_upper
            : 0.0;

    std::vector<int> ideal_group = GreedyFeasibleIdealGroup(candidate_pool_ids,
                                                            proxy_truth,
                                                            (size_t)K,
                                                            ret.min_overlap_required,
                                                            social_graph,
                                                            include_seed_itself);
    ret.ideal_spread_at_k = SpreadAtK(ideal_group, social_graph, include_seed_itself);
    ret.nspread_at_k = (ret.ideal_spread_at_k > 0)
            ? (double)ret.spread_at_k / (double)ret.ideal_spread_at_k
            : 0.0;

    return ret;
}

inline void WriteGroupTaskEvalCSV(const std::string& path,
                                  const std::vector<GroupTaskEvalResult>& rows) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::cerr << "[WriteGroupTaskEvalCSV] Cannot open file: " << path << std::endl;
        return;
    }

    fout << "query_id,method,constraint_recall,overlap_hits,min_overlap_required,"
         << "positives_hit,precision_at_k,recall_at_k,hitrate_at_k,ndcg_at_k,"
         << "spread_at_k,ideal_spread_at_k,nspread_at_k,pool_spread_upper,pool_cover_rate\n";

    fout << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        fout << r.query_id << ','
             << r.method << ','
             << r.constraint_recall << ','
             << r.overlap_hits << ','
             << r.min_overlap_required << ','
             << r.positives_hit << ','
             << r.precision_at_k << ','
             << r.recall_at_k << ','
             << r.hitrate_at_k << ','
             << r.ndcg_at_k << ','
             << r.spread_at_k << ','
             << r.ideal_spread_at_k << ','
             << r.nspread_at_k << ','
             << r.pool_spread_upper << ','
             << r.pool_cover_rate << '\n';
    }
}

inline void PrintGroupTaskEvalSummary(const std::vector<GroupTaskEvalResult>& rows) {
    if (rows.empty()) return;

    double sum_constraint_recall = 0.0;
    double sum_precision = 0.0;
    double sum_recall = 0.0;
    double sum_hr = 0.0;
    double sum_ndcg = 0.0;
    double sum_nspread = 0.0;
    double sum_pool_cover = 0.0;

    for (const auto& r : rows) {
        sum_constraint_recall += r.constraint_recall;
        sum_precision += r.precision_at_k;
        sum_recall += r.recall_at_k;
        sum_hr += r.hitrate_at_k;
        sum_ndcg += r.ndcg_at_k;
        sum_nspread += r.nspread_at_k;
        sum_pool_cover += r.pool_cover_rate;
    }

    const double n = (double)rows.size();
    std::cout << "\n========== Group Evaluation Summary ==========" << std::endl;
    std::cout << "#queries            = " << rows.size() << std::endl;
    std::cout << "Avg ConstraintRecall= " << (sum_constraint_recall / n) << std::endl;
    std::cout << "Avg Precision@K     = " << (sum_precision / n) << std::endl;
    std::cout << "Avg Recall@K        = " << (sum_recall / n) << std::endl;
    std::cout << "Avg HitRate@K       = " << (sum_hr / n) << std::endl;
    std::cout << "Avg NDCG@K          = " << (sum_ndcg / n) << std::endl;
    std::cout << "Avg NSpread@K       = " << (sum_nspread / n) << std::endl;
    std::cout << "Avg PoolCoverRate   = " << (sum_pool_cover / n) << std::endl;
    std::cout << "=============================================" << std::endl;
}

#endif // GROUP_TASK_EVAL_H
