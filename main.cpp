// main.cpp
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <queue>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <chrono>
#include <numeric>
#include <random>
#include <iomanip>
#include "MySolution.h"
using namespace std;

using fpair = pair<float,int>;
const string ROOT = "./data_o/"; // 修改根目录
const string GT_FILENAME = "groundtruth_top10.txt";
const string PRED_FILENAME = "pred_top10.txt";

const bool USE_SIFT = true;
const int QUERY_NUM = 100;
const int TOPK = 10;
const long long base_limit = -1; // -1 表示使用全部 pointsnum，否则使用前 base_limit 条作为库与 GT 计算

int dimension_, pointsnum;
string base_path;

vector<float> base_data;                 // 扁平存储
vector<vector<float>> base_vecs;         // 二维向量库 (可能是部分库)
vector<vector<float>> query_data;        // 查询向量

// 读取扁平 base.txt
void load_base() {
    ifstream fin(base_path);
    if (!fin.is_open()) {
        cerr << "Error opening base file: " << base_path << "\n";
        exit(1);
    }
    long long total = (long long)dimension_ * pointsnum;
    base_data.resize(total);
    for (long long i = 0; i < total; ++i) fin >> base_data[i];
    fin.close();
}

// 把扁平底库恢复成二维，最多 use_limit 个（若 use_limit==-1 使用全部）
void build_base_vectors(long long use_limit = -1) {
    long long N = pointsnum;
    if (use_limit > 0 && use_limit < N) N = use_limit;
    base_vecs.assign((size_t)N, vector<float>(dimension_));
    for (long long i = 0; i < N; ++i) {
        long long offset = i * (long long)dimension_;
        for (int d = 0; d < dimension_; ++d) base_vecs[i][d] = base_data[offset + d];
    }
}

// 取底库前 Q 个向量作为 query（真实查询）
void take_queries_from_base(int Q) {
    int N = (int)base_vecs.size();
    Q = min(Q, N);
    query_data.assign(Q, vector<float>(dimension_));
    for (int i = 0; i < Q; ++i) query_data[i] = base_vecs[i];
}

// L2 距离（平方）
inline float l2_sq(const vector<float>& a, const vector<float>& b) {
    float s = 0.0f;
    int D = (int)a.size();
    for (int i = 0; i < D; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

// 暴力构建 ground truth top-K
vector<array<int, TOPK>> brute_force_gt(const vector<vector<float>>& queries,
                                        const vector<vector<float>>& base,
                                        int K) {
    int Q = (int)queries.size();
    int N = (int)base.size();
    vector<array<int, TOPK>> gt(Q);
    vector<fpair> buf;
    buf.reserve(N);
    for (int qi = 0; qi < Q; ++qi) {
        buf.clear();
        const auto &q = queries[qi];
        for (int i = 0; i < N; ++i) {
            float dist = l2_sq(q, base[i]);
            buf.emplace_back(dist, i);
        }
        if ((int)buf.size() <= K) {
            sort(buf.begin(), buf.end());
            for (int k = 0; k < (int)buf.size() && k < K; ++k) gt[qi][k] = buf[k].second;
            for (int k = (int)buf.size(); k < K; ++k) gt[qi][k] = -1;
        } else {
            nth_element(buf.begin(), buf.begin()+K, buf.end(),
                        [](const fpair &a, const fpair &b){ return a.first < b.first; });
            sort(buf.begin(), buf.begin()+K, [](const fpair &a, const fpair &b){ return a.first < b.first; });
            for (int k = 0; k < K; ++k) gt[qi][k] = buf[k].second;
        }
    }
    return gt;
}

// 将 gt 保存到文本文件，每行 TOPK 个 id，以空格分隔
void save_gt_to_file(const string &fname, const vector<array<int, TOPK>>& gt) {
    ofstream fout(fname);
    if (!fout.is_open()) { cerr << "Cannot open " << fname << " for write\n"; return; }
    for (size_t i = 0; i < gt.size(); ++i) {
        for (int k = 0; k < TOPK; ++k) {
            if (k) fout << ' ';
            fout << gt[i][k];
        }
        fout << '\n';
    }
    fout.close();
}

// 从文件加载 gt，如果文件不存在或格式不对返回 empty vector
vector<array<int, TOPK>> load_gt_from_file(const string &fname) {
    ifstream fin(fname);
    vector<array<int, TOPK>> gt;
    if (!fin.is_open()) return gt;
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        array<int, TOPK> arr;
        for (int k = 0; k < TOPK; ++k) {
            if (!(iss >> arr[k])) { arr[k] = -1; }
        }
        gt.push_back(arr);
    }
    fin.close();
    return gt;
}

// 保存 pred 到文件（你的算法输出）
void save_pred_to_file(const string &fname, const vector<array<int, TOPK>>& pred) {
    ofstream fout(fname);
    if (!fout.is_open()) { cerr << "Cannot open " << fname << " for write\n"; return; }
    for (size_t i = 0; i < pred.size(); ++i) {
        for (int k = 0; k < TOPK; ++k) {
            if (k) fout << ' ';
            fout << pred[i][k];
        }
        fout << '\n';
    }
    fout.close();
}

// 计算 Recall@K
double recall_at_k(const vector<array<int, TOPK>>& gt, const vector<array<int, TOPK>>& pred) {
    if (gt.empty()) return -1.0;
    int Q = (int)gt.size();
    double sum = 0.0;
    for (int i = 0; i < Q; ++i) {
        unordered_set<int> S;
        S.reserve(TOPK*2);
        for (int k = 0; k < TOPK; ++k) if (gt[i][k] >= 0) S.insert(gt[i][k]);
        int hit = 0;
        for (int k = 0; k < TOPK; ++k) if (S.find(pred[i][k]) != S.end()) ++hit;
        sum += (double)hit / (double)TOPK;
    }
    return sum / Q;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (USE_SIFT) {
        dimension_ = 100;
        pointsnum = 1183514;
        base_path = ROOT + "glove/base.txt";
    } else {
        cerr << "Only SIFT enabled in this test program.\n";
        return 1;
    }

    cout << "Base file: " << base_path << "\n";
    cout << "Dimension = " << dimension_ << ", pointsnum = " << pointsnum << "\n";

    // 1) load base (flat)
    cout << "Loading base (flat) ...\n";
    load_base();

    // 2) build base vecs (maybe subset)
    long long useN = (base_limit > 0 ? min<long long>(base_limit, pointsnum) : pointsnum);
    cout << "Building base vectors (useN = " << useN << ") ...\n";
    build_base_vectors(useN);

    // 3) prepare queries (take first QUERY_NUM from base)
    int Q = QUERY_NUM;
    take_queries_from_base(Q);
    cout << "Using " << Q << " queries (from base prefix)\n";

    // 4) try loading GT from file
    auto gt = load_gt_from_file(GT_FILENAME);
    if (!gt.empty()) {
        if ((int)gt.size() != Q) {
            cout << "GT file exists but size mismatch (file Q=" << gt.size() << ", needed Q=" << Q << "). Recomputing GT.\n";
            gt.clear();
        } else {
            cout << "Loaded ground truth from file '" << GT_FILENAME << "' (Q=" << gt.size() << ").\n";
        }
    }

    // 5) compute brute-force GT if not loaded
    if (gt.empty()) {
        cout << "Computing brute-force ground truth (Top-" << TOPK << ") ...\n";
        auto t0 = chrono::high_resolution_clock::now();
        gt = brute_force_gt(query_data, base_vecs, TOPK);
        auto t1 = chrono::high_resolution_clock::now();
        double brutetime = chrono::duration<double>(t1 - t0).count();
        cout << "Brute-force GT time = " << brutetime << " sec\n";
        // save to disk
        cout << "Saving GT to '" << GT_FILENAME << "' ...\n";
        save_gt_to_file(GT_FILENAME, gt);
    }

    // 6) call Solution.build()
    Solution sol;

    // ---- make sure build does NOT contribute to distance comparison statistics ----
    Solution::enable_dist_counter(false);
    Solution::reset_dist_counter();
    // ---------------------------------------------------------------------------

    cout << "Calling Solution::build ...\n";
    auto tb1 = chrono::high_resolution_clock::now();
    sol.build(dimension_, base_data);
    auto tb2 = chrono::high_resolution_clock::now();
    cout << "Build time (your build) = " << chrono::duration<double>(tb2 - tb1).count() << " sec\n";

    // 7) call Solution::search on queries (more stable timing)
    cout << "Running Solution::search on queries ...\n";
    vector<array<int, TOPK>> pred(Q);
    int res[TOPK];

    // Warmup + multi-round measurement to reduce noise
    constexpr int WARMUP_ROUNDS  = 3;
    constexpr int MEASURE_ROUNDS = 10;

    // warmup (not recorded, and NOT counted)
    Solution::enable_dist_counter(false);
    for (int r = 0; r < WARMUP_ROUNDS; ++r) {
        for (int qi = 0; qi < Q; ++qi) {
            sol.search(query_data[qi], res);
        }
    }

    // reset counter after warmup, start counting for measurement only
    Solution::reset_dist_counter();
    Solution::enable_dist_counter(true);

    vector<double> per_query_ms;
    per_query_ms.reserve((size_t)Q * MEASURE_ROUNDS);

    double total_search_time = 0.0;

    for (int r = 0; r < MEASURE_ROUNDS; ++r) {
        for (int qi = 0; qi < Q; ++qi) {
            auto ts = chrono::high_resolution_clock::now();
            sol.search(query_data[qi], res);
            auto te = chrono::high_resolution_clock::now();
            double use = chrono::duration<double>(te - ts).count();

            total_search_time += use;
            per_query_ms.push_back(use * 1000.0);

            // record predictions only in the last measurement round
            if (r == MEASURE_ROUNDS - 1) {
                for (int k = 0; k < TOPK; ++k) pred[qi][k] = res[k];
            }
        }
    }

    // stop counting immediately after measurement
    Solution::enable_dist_counter(false);

    double avg_ms = (total_search_time / (double)(Q * MEASURE_ROUNDS)) * 1000.0;

    double median_ms = -1.0;
    if (!per_query_ms.empty()) {
        auto mid = per_query_ms.begin() + per_query_ms.size() / 2;
        nth_element(per_query_ms.begin(), mid, per_query_ms.end());
        median_ms = *mid;
    }

    cout << "Average search time (your search) = " << avg_ms << " ms/query\n";
    cout << "Median  search time (your search) = " << median_ms << " ms/query\n";

    // Avg distance comparisons (search only, exclude warmup/build)
    std::uint64_t total_cmp = Solution::get_dist_counter();
    double avg_cmp = (double)total_cmp / (double)(Q * MEASURE_ROUNDS);
    cout << "AvgDistComparisonsPerQuery = " << fixed << setprecision(3) << avg_cmp << "\n";

    // 8) save pred to file
    cout << "Saving predictions to '" << PRED_FILENAME << "' ...\n";
    save_pred_to_file(PRED_FILENAME, pred);

    // 9) compute recall@10
    double R10 = recall_at_k(gt, pred);
    cout << fixed << setprecision(6) << "Recall@10 = " << R10 << "\n";

    // 10) print example
    cout << "\nExample (query 0):\nGT: ";
    for (int k = 0; k < TOPK; ++k) cout << gt[0][k] << " ";
    cout << "\nPred: ";
    for (int k = 0; k < TOPK; ++k) cout << pred[0][k] << " ";
    cout << "\n";

    return 0;
}
