#include "MySolution.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>

#include <immintrin.h> // AVX2 / AVX-512 等

// 预取指令宏
#define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)

// LID 估计
static float estimate_lid_from_dists(std::vector<float>& dists) {
    if (dists.size() < 2) return 0.0f;
    std::sort(dists.begin(), dists.end());
    float rk = dists.back();
    if (rk <= 0.0f) rk = 1e-6f;

    double s = 0.0;
    const double eps = 1e-12;
    for (size_t i = 0; i + 1 < dists.size(); ++i) {
        double ri = std::max<double>(dists[i], eps);
        s += std::log(ri / (double)rk);
    }
    if (s >= -eps) return 0.0f;
    double lid = - (double)(dists.size() - 1) / s;
    if (!std::isfinite(lid)) return 0.0f;
    return (float)lid;
}

// ======================= 距离计数器 static 定义 =======================
std::atomic<bool> Solution::dist_counter_enabled_{false};
std::atomic<std::uint64_t> Solution::dist_counter_{0};

void Solution::enable_dist_counter(bool on) {
    dist_counter_enabled_.store(on, std::memory_order_relaxed);
}
void Solution::reset_dist_counter() {
    dist_counter_.store(0, std::memory_order_relaxed);
}
std::uint64_t Solution::get_dist_counter() {
    return dist_counter_.load(std::memory_order_relaxed);
}

// ======================= 构造函数 =======================

Solution::Solution() {
    rng_.seed(42u);
}

// ======================= 量化相关 =======================

void Solution::quantize_vec(const float* src, int16_t* dst) const {
    for (int i = 0; i < dim_; ++i) {
        float v = (src[i] - min_val_) * diff_scale_;
        if (v < -32000.0f) v = -32000.0f;
        if (v >  32000.0f) v =  32000.0f;
        dst[i] = (int16_t)std::lrintf(v);
    }
    for (int i = dim_; i < dim_aligned_; ++i) dst[i] = 0;
}

// AVX2 版本的 SQ16 L2 距离，如果没有 AVX2 就退化为标量
float Solution::dist_sq16(const int16_t* d1, const int16_t* d2) const {
    inc_dist_counter_();

#ifdef __AVX2__
    int d = dim_aligned_;
    __m256i sum_i32 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(d1 + i));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(d2 + i));
        __m256i diff = _mm256_sub_epi16(v1, v2);
        __m256i sq_sum = _mm256_madd_epi16(diff, diff);
        sum_i32 = _mm256_add_epi32(sum_i32, sq_sum);
    }

    int32_t tail_sum = 0;
    for (; i < d; ++i) {
        int v = (int)d1[i] - (int)d2[i];
        tail_sum += v * v;
    }

    __m128i sum_low  = _mm256_castsi256_si128(sum_i32);
    __m128i sum_high = _mm256_extracti128_si256(sum_i32, 1);
    __m128i sum128   = _mm_add_epi32(sum_low, sum_high);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1,0,3,2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0,3,0,1)));
    int32_t res_int = _mm_cvtsi128_si32(sum128);
    res_int += tail_sum;

    return (float)res_int;
#else
    int d = dim_aligned_;
    int32_t s = 0;
    for (int i = 0; i < d; ++i) {
        int v = (int)d1[i] - (int)d2[i];
        s += v * v;
    }
    return (float)s;
#endif
}

float Solution::dist2_float(const float* a, const float* b) const {
    inc_dist_counter_();
    float s = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// ======================= HNSW 级别相关 =======================

int Solution::sampleLevel() {
    static const float ml = 1.0f / std::log((float)M_);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng_);
    if (u < 1e-9f) u = 1e-9f;
    return (int)(-std::log(u) * ml);
}

// ----------------- Greedy Search (Top Layers) -----------------

int Solution::greedy_search_layer(const int16_t* q_sq, int ep, int level) const {
    IdType cur = ep;
    float curDist = dist_sq16(q_sq, getVecSQ(cur));
    bool changed = true;

    while (changed) {
        changed = false;
        const auto& neigh = graph_[level][cur];

        if (!neigh.empty()) PREFETCH(getVecSQ(neigh[0]));

        for (std::size_t i = 0; i < neigh.size(); ++i) {
            int nb = neigh[i];
            if (i + 1 < neigh.size()) PREFETCH(getVecSQ(neigh[i + 1]));
            float d = dist_sq16(q_sq, getVecSQ(nb));
            if (d < curDist) {
                curDist = d;
                cur = nb;
                changed = true;
            }
        }
    }
    return cur;
}

// ----------------- Beam Search (Bottom Layer, Search API) -----------------

std::vector<Solution::Pair> Solution::search_layer_with_dist(
    const int16_t* q_sq, int ep, int K, int level) const
{
    if (K > N_) K = N_;

    auto cmpMin = [](const Pair& a, const Pair& b) { return a.first > b.first; };
    auto cmpMax = [](const Pair& a, const Pair& b) { return a.first < b.first; };

    std::priority_queue<Pair, std::vector<Pair>, decltype(cmpMin)> cand(cmpMin);
    std::priority_queue<Pair, std::vector<Pair>, decltype(cmpMax)> Bk(cmpMax);

    const float gamma = gamma_;

    if (visited_.empty()) {
        visited_.assign(N_, 0u);
        visitedToken_ = 1u;
    }
    if (++visitedToken_ == 0u) {
        std::fill(visited_.begin(), visited_.end(), 0u);
        visitedToken_ = 1u;
    }

    visited_[ep] = visitedToken_;
    float dist_ep = dist_sq16(q_sq, getVecSQ(ep));
    cand.push({dist_ep, ep});
    Bk.push({dist_ep, ep});

    while (!cand.empty()) {
        Pair curPair = cand.top();
        cand.pop();

        float dist_k = Bk.top().first;
        float bound  = (1.0f + gamma) * dist_k;

        if (curPair.first > bound) break;

        IdType curId = curPair.second;

        const int* neighbors_ptr = nullptr;
        int size = 0;

        if (level == 0 && !flat_graph_l0_.empty()) {
            std::size_t offset = (std::size_t)curId * (M0_ + 1);
            size = flat_graph_l0_[offset];
            neighbors_ptr = &flat_graph_l0_[offset + 1];
        } else {
            const auto& vec = graph_[level][curId];
            neighbors_ptr = vec.data();
            size = (int)vec.size();
        }

        if (size > 0) PREFETCH(getVecSQ(neighbors_ptr[0]));

        for (int i = 0; i < size; ++i) {
            int nb = neighbors_ptr[i];

            if (i + 1 < size) PREFETCH(getVecSQ(neighbors_ptr[i + 1]));

            if (visited_[nb] == visitedToken_) continue;
            visited_[nb] = visitedToken_;

            float d = dist_sq16(q_sq, getVecSQ(nb));

            if ((int)Bk.size() < K || d < Bk.top().first) {
                Bk.push({d, nb});
                if ((int)Bk.size() > K) Bk.pop();
            }

            dist_k = Bk.top().first;
            bound  = (1.0f + gamma) * dist_k;

            if (d <= bound) cand.push({d, nb});
        }
    }

    std::vector<Pair> result;
    result.reserve(Bk.size());
    while (!Bk.empty()) {
        result.push_back(Bk.top());
        Bk.pop();
    }
    std::sort(result.begin(), result.end(),
              [](const Pair& a, const Pair& b){ return a.first < b.first; });
    return result;
}

// ----------------- 选择邻居 Heuristic (RNG 风格) -----------------

void Solution::select_neighbors_heuristic(std::vector<Pair>& candidates,
                                          int M,
                                          std::vector<int>& out) const
{
    out.clear();
    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(),
              [](const Pair& a, const Pair& b) { return a.first < b.first; });

    out.reserve(M);
    for (const auto& c : candidates) {
        if ((int)out.size() >= M) break;
        int v = c.second;
        bool ok = true;
        const int16_t* vVec = getVecSQ(v);
        for (int u : out) {
            float d = dist_sq16(getVecSQ(u), vVec);
            if (d < c.first) { ok = false; break; }
        }
        if (ok) out.push_back(v);
    }
}

// ======================= Build Process =======================

void Solution::build(int dim, const std::vector<float>& base_flat) {
    dim_ = dim;
    N_   = (int)(base_flat.size() / dim_);
    if (N_ <= 0) return;

    // 1) 量化准备
    dim_aligned_ = (dim_ + 15) / 16 * 16;
    data_sq_.assign((std::size_t)N_ * dim_aligned_, 0);

    min_val_ = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (float x : base_flat) {
        if (x < min_val_) min_val_ = x;
        if (x > max_val)  max_val = x;
    }
    if (max_val == min_val_) max_val += 1e-6f;
    float range = max_val - min_val_;
    diff_scale_ = 30000.0f / range;

    for (int i = 0; i < N_; ++i) {
        const float* src = &base_flat[(std::size_t)i * dim_];
        int16_t* dst     = &data_sq_[(std::size_t)i * dim_aligned_];
        quantize_vec(src, dst);
    }

    // 2) 先采样每个点的层数（两次建图复用同一份 nodeLevel_，避免随机噪声影响“插入顺序实验”）
    nodeLevel_.assign(N_, 0);
    int globalMax = 0;
    for (int i = 0; i < N_; ++i) {
        int l = sampleLevel();
        nodeLevel_[i] = l;
        if (l > globalMax) globalMax = l;
    }

    auto init_graph_and_locks = [&]() {
        graph_.clear();
        graph_.resize(globalMax + 1);
        for (int l = 0; l <= globalMax; ++l) graph_[l].resize(N_);

        nodeLocks_.resize(N_);
        for (int i = 0; i < N_; ++i) nodeLocks_[i] = std::make_unique<std::mutex>();

        maxLevel_   = -1;
        enterPoint_ = -1;
        flat_graph_l0_.clear();
    };

    auto parallel_insert_by_order = [&](const std::vector<int>& buildOrder) {
        std::atomic<int> nextIdx(0);
        int nThreads = (int)std::thread::hardware_concurrency();
        if (nThreads <= 0) nThreads = 4;

        std::vector<std::thread> threads;
        auto worker = [&]() {
            std::vector<std::uint32_t> visited_local;
            std::uint32_t token_local = 1u;
            while (true) {
                int idx = nextIdx.fetch_add(1);
                if (idx >= (int)buildOrder.size()) break;
                int u = buildOrder[idx];
                addPointMT(u, getVecSQ(u), visited_local, token_local);
            }
        };

        for (int t = 0; t < nThreads; ++t) threads.emplace_back(worker);
        for (auto& th : threads) th.join();
    };

    // 3) 默认插入顺序（按第0维投影升序）
    std::vector<int> order_proj(N_);
    for (int i = 0; i < N_; ++i) order_proj[i] = i;
    std::sort(order_proj.begin(), order_proj.end(),
              [&](int a, int b) {
                  return base_flat[(std::size_t)a * dim_] < base_flat[(std::size_t)b * dim_];
              });

    std::vector<float> lid;

    // ==================== 预处理开关：从 h 的 bool 控制 ====================
    if (enable_lid_preprocess_) {
        // Pass 1：粗建图估 LID
        int efC_save = efC_;
        efC_ = EF_LID_BUILD_;

        init_graph_and_locks();
        parallel_insert_by_order(order_proj);

        lid.assign(N_, 0.0f);
        for (int i = 0; i < N_; ++i) {
            auto& nb = graph_[0][i];
            if (nb.empty()) { lid[i] = 0.0f; continue; }
            int take = std::min<int>((int)nb.size(), K_LID_);

            std::vector<float> d;
            d.reserve((size_t)take);
            const int16_t* vi = getVecSQ(i);

            for (int t = 0; t < take; ++t) {
                int j = nb[t];
                if (j < 0 || j >= N_ || j == i) continue;
                float dij = dist_sq16(vi, getVecSQ(j));
                if (dij <= 0.0f) dij = 1e-6f;
                d.push_back(dij);
            }
            lid[i] = estimate_lid_from_dists(d);
        }

        efC_ = efC_save;

        // Pass 2：按 LID 降序重建（正式图）
        std::vector<int> order_lid(N_);
        for (int i = 0; i < N_; ++i) order_lid[i] = i;
        std::sort(order_lid.begin(), order_lid.end(),
                  [&](int a, int b) { return lid[a] > lid[b]; });

        init_graph_and_locks();
        parallel_insert_by_order(order_lid);

    } else {
        // 单次建图（原逻辑）
        init_graph_and_locks();
        parallel_insert_by_order(order_proj);
    }

    // ======================= Post-process: reverse-edge injection (Level 0) =======================
    auto prune_level0 = [&](int v) {
        auto& nbv = graph_[0][v];

        int w = 0;
        for (int x : nbv) if (x >= 0 && x < N_ && x != v) nbv[w++] = x;
        nbv.resize(w);

        if (!nbv.empty()) {
            std::sort(nbv.begin(), nbv.end());
            nbv.erase(std::unique(nbv.begin(), nbv.end()), nbv.end());
        }

        if ((int)nbv.size() <= M0_) return;

        std::vector<Pair> cand;
        cand.reserve(nbv.size());
        const int16_t* vVec = getVecSQ(v);
        for (int other : nbv) {
            if (other < 0 || other >= N_ || other == v) continue;
            float d = dist_sq16(getVecSQ(other), vVec);
            cand.emplace_back(d, other);
        }
        std::vector<int> pruned;
        select_neighbors_heuristic(cand, M0_, pruned);
        nbv.assign(pruned.begin(), pruned.end());

        w = 0;
        for (int x : nbv) if (x >= 0 && x < N_ && x != v) nbv[w++] = x;
        nbv.resize(w);

        if (!nbv.empty()) {
            std::sort(nbv.begin(), nbv.end());
            nbv.erase(std::unique(nbv.begin(), nbv.end()), nbv.end());
        }
    };

    {
        const int R = 1;
        for (int round = 0; round < R; ++round) {
            for (int u = 0; u < N_; ++u) {
                std::vector<int> tmp = graph_[0][u];
                for (int v : tmp) {
                    if (v < 0 || v >= N_ || v == u) continue;
                    auto& nbv = graph_[0][v];
                    if (std::find(nbv.begin(), nbv.end(), u) == nbv.end()) {
                        nbv.push_back(u);
                        if ((int)nbv.size() > M0_) prune_level0(v);
                    }
                }
            }
        }
        for (int v = 0; v < N_; ++v) prune_level0(v);
    }

    // 6) 扁平化 Level 0
    int stride = M0_ + 1;
    flat_graph_l0_.assign((std::size_t)N_ * (std::size_t)stride, -1);
    for (int i = 0; i < N_; ++i) {
        const auto& nb = graph_[0][i];
        std::size_t off = (std::size_t)i * (std::size_t)stride;

        int count = 0;
        for (int x : nb) {
            if (count < M0_) flat_graph_l0_[off + 1 + count++] = x;
        }
        flat_graph_l0_[off] = count;
    }
}

// ----------------- 构建时的 search_layer_mt（线程本地 visited） -----------------

void Solution::search_layer_mt(const int16_t* q_sq,
                               int ep,
                               int ef,
                               int level,
                               std::priority_queue<Pair>& topRes,
                               std::vector<std::uint32_t>& visited_local,
                               std::uint32_t& visitedTokenLocal)
{
    if (ef > N_) ef = N_;

    auto cmpMin = [](const Pair& a, const Pair& b) { return a.first > b.first; };

    std::priority_queue<Pair, std::vector<Pair>, decltype(cmpMin)> cand(cmpMin);
    topRes = std::priority_queue<Pair>();

    if ((int)visited_local.size() != N_) {
        visited_local.assign(N_, 0u);
        visitedTokenLocal = 1u;
    }
    if (++visitedTokenLocal == 0u) {
        std::fill(visited_local.begin(), visited_local.end(), 0u);
        visitedTokenLocal = 1u;
    }

    visited_local[ep] = visitedTokenLocal;
    float dist_ep = dist_sq16(q_sq, getVecSQ(ep));
    cand.push({dist_ep, ep});
    topRes.push({dist_ep, ep});
    DistType topResLimit = dist_ep;

    while (!cand.empty()) {
        Pair curPair = cand.top();
        cand.pop();

        if (curPair.first > topResLimit) break;
        IdType curId = curPair.second;

        std::vector<int> neighbors;
        {
            std::lock_guard<std::mutex> lk(*nodeLocks_[curId]);
            neighbors = graph_[level][curId];
        }

        if (!neighbors.empty()) PREFETCH(getVecSQ(neighbors[0]));

        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            int nb = neighbors[i];
            if (i + 1 < neighbors.size()) PREFETCH(getVecSQ(neighbors[i + 1]));

            if (visited_local[nb] == visitedTokenLocal) continue;
            visited_local[nb] = visitedTokenLocal;

            float d = dist_sq16(q_sq, getVecSQ(nb));

            if ((int)topRes.size() < ef || d < topResLimit) {
                cand.push({d, nb});
                topRes.push({d, nb});
                if ((int)topRes.size() > ef) topRes.pop();
                topResLimit = topRes.top().first;
            }
        }
    }
}

// ----------------- addPointMT：并行插入一个节点 -----------------

void Solution::addPointMT(int cur,
                          const int16_t* curVecSQ,
                          std::vector<std::uint32_t>& visited_local,
                          std::uint32_t& visitedTokenLocal)
{
    int level = nodeLevel_[cur];

    {
        std::lock_guard<std::mutex> g(globalLock_);
        if (enterPoint_ < 0 || maxLevel_ < 0) {
            enterPoint_ = cur;
            maxLevel_   = level;
            return;
        }
    }

    int curMaxLevel;
    int ep;
    {
        std::lock_guard<std::mutex> g(globalLock_);
        ep = enterPoint_;
        curMaxLevel = maxLevel_;
    }

    if (level > curMaxLevel) {
        std::lock_guard<std::mutex> g(globalLock_);
        if (level > maxLevel_) {
            maxLevel_   = level;
            enterPoint_ = cur;
        }
        curMaxLevel = maxLevel_;
    }

    const int16_t* q_sq = curVecSQ;
    for (int l = curMaxLevel; l > level; --l) {
        ep = greedy_search_layer(q_sq, ep, l);
    }

    for (int l = std::min(level, curMaxLevel); l >= 0; --l) {
        std::priority_queue<Pair> topRes;
        search_layer_mt(q_sq, ep, efC_, l, topRes, visited_local, visitedTokenLocal);

        std::vector<Pair> layer_cand;
        while (!topRes.empty()) {
            layer_cand.push_back(topRes.top());
            topRes.pop();
        }
        std::sort(layer_cand.begin(), layer_cand.end(),
                  [](const Pair& a, const Pair& b) { return a.first < b.first; });

        std::vector<int> selected;
        int maxM = maxMForLevel(l);
        select_neighbors_heuristic(layer_cand, maxM, selected);

        selected.erase(
            std::remove_if(selected.begin(), selected.end(),
                [&](int x){ return x < 0 || x >= N_ || x == cur; }),
            selected.end());

        {
            std::lock_guard<std::mutex> lk(*nodeLocks_[cur]);
            auto& neigh = graph_[l][cur];
            neigh.insert(neigh.end(), selected.begin(), selected.end());

            if (!neigh.empty()) {
                std::sort(neigh.begin(), neigh.end());
                neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());
            }

            int maxM2 = maxMForLevel(l);
            if ((int)neigh.size() > maxM2) {
                std::vector<Pair> cand_cur;
                cand_cur.reserve(neigh.size());
                const int16_t* curV = getVecSQ(cur);
                for (int other : neigh) {
                    if (other < 0 || other >= N_ || other == cur) continue;
                    float d = dist_sq16(getVecSQ(other), curV);
                    cand_cur.emplace_back(d, other);
                }
                std::vector<int> pruned;
                select_neighbors_heuristic(cand_cur, maxM2, pruned);
                neigh.assign(pruned.begin(), pruned.end());

                neigh.erase(std::remove_if(neigh.begin(), neigh.end(),
                    [&](int x){ return x < 0 || x >= N_ || x == cur; }),
                    neigh.end());
            }
        }

        for (int nb : selected) {
            if (nb < 0 || nb >= N_ || nb == cur) continue;
            {
                std::lock_guard<std::mutex> lk(*nodeLocks_[nb]);
                auto& neigh_nb = graph_[l][nb];
                neigh_nb.push_back(cur);

                if ((int)neigh_nb.size() > maxMForLevel(l)) {
                    std::vector<Pair> cand_nb;
                    cand_nb.reserve(neigh_nb.size());
                    const int16_t* nbVec = getVecSQ(nb);
                    for (int other : neigh_nb) {
                        if (other < 0 || other >= N_ || other == nb) continue;
                        float d = dist_sq16(getVecSQ(other), nbVec);
                        cand_nb.emplace_back(d, other);
                    }
                    std::vector<int> pruned;
                    select_neighbors_heuristic(cand_nb, maxMForLevel(l), pruned);
                    neigh_nb.assign(pruned.begin(), pruned.end());

                    neigh_nb.erase(std::remove_if(neigh_nb.begin(), neigh_nb.end(),
                        [&](int x){ return x < 0 || x >= N_ || x == nb; }),
                        neigh_nb.end());
                }
            }
        }

        if (!selected.empty()) ep = selected[0];
    }
}

// ======================= Search API =======================

void Solution::search(const std::vector<float>& q, int* res) {
    if (N_ == 0) {
        for (int i = 0; i < 10; ++i) res[i] = 0;
        return;
    }

    std::vector<int16_t> q_sq(dim_aligned_);
    quantize_vec(q.data(), q_sq.data());
    const int16_t* qPtr = q_sq.data();

    int ep;
    int curMax;
    {
        std::lock_guard<std::mutex> g(globalLock_);
        ep = enterPoint_;
        curMax = maxLevel_;
    }
    for (int l = curMax; l > 0; --l) ep = greedy_search_layer(qPtr, ep, l);

    const int ef = (efS_ > 0 ? efS_ : 10);
    std::vector<Pair> cand = search_layer_with_dist(qPtr, ep, ef, 0);

    for (int i = 0; i < 10; ++i) {
        res[i] = (i < (int)cand.size()) ? cand[i].second : 0;
    }
}
