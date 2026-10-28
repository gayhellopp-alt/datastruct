#include "MySolution.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include <immintrin.h> // AVX2 / AVX-512 等

//测量
std::atomic<std::uint64_t> Solution::dist_counter_{0};
std::atomic<bool> Solution::dist_count_enabled_{false};
void Solution::enable_dist_counter(bool on) {
    dist_count_enabled_.store(on, std::memory_order_relaxed);
}
void Solution::reset_dist_counter(){ dist_counter_.store(0, std::memory_order_relaxed); }
std::uint64_t Solution::get_dist_counter(){ return dist_counter_.load(std::memory_order_relaxed); }


// 预取指令宏
#if ABLATE_DISABLE_PREFETCH
#define PREFETCH(addr) ((void)0)
#else
#define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#endif

// ======================= 构造函数 =======================

Solution::Solution() {
    rng_.seed(42u);
}

// ======================= 量化相关 =======================

void Solution::quantize_vec(const float* src, int16_t* dst) const {
    // 简单线性量化：val_i16 = (val_f - min_val_) * diff_scale_
    for (int i = 0; i < dim_; ++i) {
        float v = (src[i] - min_val_) * diff_scale_;
        if (v < -32000.0f) v = -32000.0f;
        if (v >  32000.0f) v =  32000.0f;
        dst[i] = (int16_t)std::lrintf(v);
    }
    // 填充对齐部分为 0
    for (int i = dim_; i < dim_aligned_; ++i) {
        dst[i] = 0;
    }
}

// AVX2 版本的 SQ16 L2 距离，如果没有 AVX2 就退化为标量
float Solution::dist_sq16(const int16_t* d1, const int16_t* d2) const {
    if (dist_count_enabled_.load(std::memory_order_relaxed)) {
        dist_counter_.fetch_add(1, std::memory_order_relaxed);
    }
#ifdef __AVX2__
    int d = dim_aligned_;
    __m256i sum_i32 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(d1 + i));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(d2 + i));
        __m256i diff = _mm256_sub_epi16(v1, v2);
        // diff * diff + diff * diff（相邻 int16 对调用 madd 得到 int32）
        __m256i sq_sum = _mm256_madd_epi16(diff, diff);
        sum_i32 = _mm256_add_epi32(sum_i32, sq_sum);
    }

    // 剩余尾巴用标量处理
    int32_t tail_sum = 0;
    for (; i < d; ++i) {
        int v = (int)d1[i] - (int)d2[i];
        tail_sum += v * v;
    }

    // 水平加总 sum_i32
    __m128i sum_low  = _mm256_castsi256_si128(sum_i32);
    __m128i sum_high = _mm256_extracti128_si256(sum_i32, 1);
    __m128i sum128   = _mm_add_epi32(sum_low, sum_high);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1,0,3,2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0,3,0,1)));
    int32_t res_int = _mm_cvtsi128_si32(sum128);
    res_int += tail_sum;

    return (float)res_int;   // 只用于比较大小，不做反量纲
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
    // 备用的 float L2（目前几乎不用）
    if (dist_count_enabled_.load(std::memory_order_relaxed)) {
        dist_counter_.fetch_add(1, std::memory_order_relaxed);
    }
    float s = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// ======================= HNSW 级别相关 =======================

int Solution::sampleLevel() {
    // 标准 HNSW 指数分布
    static const float ml = 1.0f / std::log((float)M_);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng_);
    if (u < 1e-9f) u = 1e-9f;
    return (int)(-std::log(u) * ml);
}

// ----------------- Greedy Search (Top Layers) -----------------

int Solution::greedy_search_layer(const VecType* q_vec, int ep, int level) const {
    IdType cur = ep;
    float curDist = dist_vec(q_vec, getVec(cur));
    bool changed = true;

    while (changed) {
        changed = false;
        const auto& neigh = graph_[level][cur]; // 上层不扁平化
        
        if (!neigh.empty()) PREFETCH(getVec(neigh[0]));

        for (std::size_t i = 0; i < neigh.size(); ++i) {
            int nb = neigh[i];
            if (i + 1 < neigh.size()) {
                PREFETCH(getVec(neigh[i + 1]));
            }
            float d = dist_vec(q_vec, getVec(nb));
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
    const VecType* q_vec, int ep, int K, int level) const
{
    if (K > N_) K = N_;

    auto cmpMin = [](const Pair& a, const Pair& b) { return a.first > b.first; };
    auto cmpMax = [](const Pair& a, const Pair& b) { return a.first < b.first; };

    std::priority_queue<Pair, std::vector<Pair>, decltype(cmpMin)> cand(cmpMin); // 候选集 C（小根堆）
    std::priority_queue<Pair, std::vector<Pair>, decltype(cmpMax)> Bk(cmpMax);   // 答案集 Bk（大根堆）

    const float gamma = gamma_;

    // 初始化 visited
    if (visited_.empty()) {
        visited_.assign(N_, 0u);
        visitedToken_ = 1u;
    }
    if (++visitedToken_ == 0u) {
        std::fill(visited_.begin(), visited_.end(), 0u);
        visitedToken_ = 1u;
    }

    visited_[ep] = visitedToken_;
    float dist_ep = dist_vec(q_vec, getVec(ep));
    cand.push({dist_ep, ep});
    Bk.push({dist_ep, ep});

    while (!cand.empty()) {
        Pair curPair = cand.top();
        cand.pop();

        // 当前第 k 近邻的距离
        float dist_k = Bk.top().first;
        float bound  = (1.0f + gamma) * dist_k;

        // 如果 C 中弹出来的点已经大于 (1+gamma)*dist_k，可以直接停止
#if !ABLATE_DISABLE_GAMMA_EARLY_STOP
        if (curPair.first > bound) break;
#endif

        IdType curId = curPair.second;

        // 取邻居（0 层用扁平化图，其他层用 graph_）
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

        if (size > 0) PREFETCH(getVec(neighbors_ptr[0]));

        for (int i = 0; i < size; ++i) {
            int nb = neighbors_ptr[i];

            if (i + 1 < size) {
                int next_nb = neighbors_ptr[i + 1];
                PREFETCH(getVec(next_nb));
            }

            if (visited_[nb] == visitedToken_) continue;
            visited_[nb] = visitedToken_;

            float d = dist_vec(q_vec, getVec(nb));

            // 1) Bk 的答案集照常更新（top-K）
            if ((int)Bk.size() < K || d < Bk.top().first) {
                Bk.push({d, nb});
                if ((int)Bk.size() > K) {
                    Bk.pop();
                }
            }

            // 重新取新的第 k 近邻（可能刚刚被更新了）
            dist_k = Bk.top().first;
            bound  = (1.0f + gamma) * dist_k;

            // 2) 候选集 C 的更新策略：仅当 d <= (1+gamma)*dist_k 时才放入 C
#if ABLATE_CAND_ALWAYS_ENQUEUE
            if (true) {
#else
            if (d <= bound) {
#endif
                cand.push({d, nb});
            }
        }
    }

    // 把 Bk 中的结果按距离升序导出
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
        const VecType* vVec = getVec(v);
        for (int u : out) {
            float d = dist_vec(getVec(u), vVec);
            if (d < c.first) { // RNG 规则
                ok = false;
                break;
            }
        }
        if (ok) out.push_back(v);
    }
}

// ======================= Build Process =======================

void Solution::build(int dim, const std::vector<float>& base_flat) {
    if (const char* s = std::getenv("M0"))   M0_ = std::atoi(s);
    if (const char* s = std::getenv("EFC"))  efC_ = std::atoi(s);
    if (const char* s = std::getenv("GAMMA")) gamma_ = std::atof(s);
    dim_ = dim;
    N_   = (int)(base_flat.size() / dim_);
    if (N_ <= 0) return;

    // 1. 初始化量化参数
#if ABLATE_USE_FLOAT_DISTANCE
    dim_aligned_ = dim_;
    data_float_.assign(base_flat.begin(), base_flat.end());
#else
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

    // 2. 执行量化
    for (int i = 0; i < N_; ++i) {
        const float* src = &base_flat[(std::size_t)i * dim_];
        int16_t* dst     = &data_sq_[(std::size_t)i * dim_aligned_];
        quantize_vec(src, dst);
    }
#endif

    // 3. 初始化层数、graph、锁
    nodeLevel_.assign(N_, 0);
    int globalMax = 0;
    for (int i = 0; i < N_; ++i) {
        int l = sampleLevel();
        nodeLevel_[i] = l;
        if (l > globalMax) globalMax = l;
    }

    graph_.clear();
    graph_.resize(globalMax + 1);
    for (int l = 0; l <= globalMax; ++l) {
        graph_[l].resize(N_);
    }

#if !ABLATE_SERIAL_BUILD
    nodeLocks_.resize(N_);
    for (int i = 0; i < N_; ++i) {
        nodeLocks_[i] = std::make_unique<std::mutex>();
    }
#endif

    maxLevel_   = -1;
    enterPoint_ = -1;

    // 4. 排序插入（你目前用第0维投影）
    std::vector<int> buildOrder(N_);
    for (int i = 0; i < N_; ++i) buildOrder[i] = i;
    std::sort(buildOrder.begin(), buildOrder.end(),
              [&](int a, int b) {
                  return base_flat[(std::size_t)a * dim_] < base_flat[(std::size_t)b * dim_];
              });

    // 5. 插入
#if ABLATE_SERIAL_BUILD
    {
        std::vector<std::uint32_t> visited_local;
        std::uint32_t token_local = 1u;
        for (int idx = 0; idx < N_; ++idx) {
            int u = buildOrder[idx];
            const VecType* vec = getVec(u);
            addPointMT(u, vec, visited_local, token_local);
        }
    }
#else
    std::atomic<int> nextIdx(0);
    int nThreads = (int)std::thread::hardware_concurrency();
    if (nThreads <= 0) nThreads = 4;
    std::vector<std::thread> threads;

    auto worker = [&]() {
        std::vector<std::uint32_t> visited_local;
        std::uint32_t token_local = 1u;
        while (true) {
            int idx = nextIdx.fetch_add(1);
            if (idx >= N_) break;
            int u = buildOrder[idx];
            const VecType* vec = getVec(u);
            addPointMT(u, vec, visited_local, token_local);
        }
    };

    for (int t = 0; t < nThreads; ++t) threads.emplace_back(worker);
    for (auto& th : threads) th.join();
#endif

    // ======================= Post-process: reverse-edge injection (Level 0) =======================
    // build 完成后做 1~2 轮：补全 u->v 的反向边 v->u，并在超限时剪枝（MIRAGE 的 R 轮思想低成本版）
#if !ABLATE_DISABLE_POST_PROCESS
    auto prune_level0 = [&](int v) {
        auto& nbv = graph_[0][v];

        // 1) 过滤非法/自环（避免 getVecSQ 越界导致 segfault）
        int w = 0;
        for (int x : nbv) {
            if (x >= 0 && x < N_ && x != v) nbv[w++] = x;
        }
        nbv.resize(w);

        // 2) 去重
        if (!nbv.empty()) {
            std::sort(nbv.begin(), nbv.end());
            nbv.erase(std::unique(nbv.begin(), nbv.end()), nbv.end());
        }

        if ((int)nbv.size() <= M0_) return;

        // 3) RNG-style 剪枝
        std::vector<Pair> cand;
        cand.reserve(nbv.size());
        const VecType* vVec = getVec(v);
        for (int other : nbv) {
            // 再保险一次
            if (other < 0 || other >= N_ || other == v) continue;
            float d = dist_vec(getVec(other), vVec);
            cand.emplace_back(d, other);
        }
        std::vector<int> pruned;
        select_neighbors_heuristic(cand, M0_, pruned);
        nbv.assign(pruned.begin(), pruned.end());

        // 4) 剪枝后再次过滤/去重（防止 pruned 混入脏值）
        w = 0;
        for (int x : nbv) {
            if (x >= 0 && x < N_ && x != v) nbv[w++] = x;
        }
        nbv.resize(w);
        if (!nbv.empty()) {
            std::sort(nbv.begin(), nbv.end());
            nbv.erase(std::unique(nbv.begin(), nbv.end()), nbv.end());
        }
    };

    {
        const int R = 1; // 建议先 1；想更稳可以试 2（build 会更慢）
        for (int round = 0; round < R; ++round) {
            for (int u = 0; u < N_; ++u) {
                // 拷贝邻居，避免遍历时被我们对 graph_[0][v] 的操作影响
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

        // 最后全局再清理一遍（更干净、更稳）
        for (int v = 0; v < N_; ++v) prune_level0(v);
    }
#endif

    // 6. 扁平化 Level 0
    int stride = M0_ + 1;
    flat_graph_l0_.assign((std::size_t)N_ * stride, -1);
    for (int i = 0; i < N_; ++i) {
        const auto& nb = graph_[0][i];
        std::size_t off = (std::size_t)i * stride;

        int count = 0;
        for (int x : nb) {
            if (count < M0_) {
                flat_graph_l0_[off + 1 + count] = x;
                ++count;
            }
        }
        flat_graph_l0_[off] = count;
    }
}


// ----------------- 构建时的 search_layer_mt（线程本地 visited） -----------------

void Solution::search_layer_mt(const VecType* q_vec,
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
    topRes = std::priority_queue<Pair>(); // 清空

    if ((int)visited_local.size() != N_) {
        visited_local.assign(N_, 0u);
        visitedTokenLocal = 1u;
    }
    if (++visitedTokenLocal == 0u) {
        std::fill(visited_local.begin(), visited_local.end(), 0u);
        visitedTokenLocal = 1u;
    }

    visited_local[ep] = visitedTokenLocal;
    float dist_ep = dist_vec(q_vec, getVec(ep));
    cand.push({dist_ep, ep});
    topRes.push({dist_ep, ep});
    DistType topResLimit = dist_ep;

    while (!cand.empty()) {
        Pair curPair = cand.top();
        cand.pop();

        if (curPair.first > topResLimit) break;
        IdType curId = curPair.second;

        // 为了避免和其他线程竞争锁，只在拷贝邻接表时加锁
        std::vector<int> neighbors;
#if ABLATE_SERIAL_BUILD
        neighbors = graph_[level][curId];
#else
        {
            std::lock_guard<std::mutex> lk(*nodeLocks_[curId]);
            neighbors = graph_[level][curId];
        }
#endif

        if (!neighbors.empty()) PREFETCH(getVec(neighbors[0]));

        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            int nb = neighbors[i];
            if (i + 1 < neighbors.size()) {
                PREFETCH(getVec(neighbors[i + 1]));
            }

            if (visited_local[nb] == visitedTokenLocal) continue;
            visited_local[nb] = visitedTokenLocal;

            float d = dist_vec(q_vec, getVec(nb));

            if ((int)topRes.size() < ef || d < topResLimit) {
                cand.push({d, nb});
                topRes.push({d, nb});
                if ((int)topRes.size() > ef) {
                    topRes.pop();
                }
                topResLimit = topRes.top().first;
            }
        }
    }
}

// ----------------- addPointMT：并行插入一个节点 -----------------

void Solution::addPointMT(int cur,
                          const VecType* curVec,
                          std::vector<std::uint32_t>& visited_local,
                          std::uint32_t& visitedTokenLocal)
{
    int level = nodeLevel_[cur];

    // 第一个点：特殊处理
#if ABLATE_SERIAL_BUILD
    if (enterPoint_ < 0 || maxLevel_ < 0) {
        enterPoint_ = cur;
        maxLevel_   = level;
        return;
    }
#else
    {
        std::lock_guard<std::mutex> g(globalLock_);
        if (enterPoint_ < 0 || maxLevel_ < 0) {
            enterPoint_ = cur;
            maxLevel_   = level;
            return;
        }
    }
#endif

    int curMaxLevel;
    int ep;
#if ABLATE_SERIAL_BUILD
    ep = enterPoint_;
    curMaxLevel = maxLevel_;
#else
    {
        std::lock_guard<std::mutex> g(globalLock_);
        ep = enterPoint_;
        curMaxLevel = maxLevel_;
    }
#endif

    // 如果当前点层数超过全局最大层，更新 enterPoint_ / maxLevel_
    if (level > curMaxLevel) {
#if ABLATE_SERIAL_BUILD
        if (level > maxLevel_) {
            maxLevel_   = level;
            enterPoint_ = cur;
        }
        curMaxLevel = maxLevel_;
#else
        std::lock_guard<std::mutex> g(globalLock_);
        if (level > maxLevel_) {
            maxLevel_   = level;
            enterPoint_ = cur;
        }
        curMaxLevel = maxLevel_;
#endif
    }

    // 1) 从最高层到 level+1 做 greedy search
    const VecType* q_vec = curVec;
    for (int l = curMaxLevel; l > level; --l) {
        ep = greedy_search_layer(q_vec, ep, l);
    }

    // 2) 从 min(level, curMaxLevel) 往下到 0 层，做多层连接
    for (int l = std::min(level, curMaxLevel); l >= 0; --l) {
        std::priority_queue<Pair> topRes;
        search_layer_mt(q_vec, ep, efC_, l, topRes, visited_local, visitedTokenLocal);

        // 将 topRes 转为 vector，并按距离升序排序
        std::vector<Pair> layer_cand;
        while (!topRes.empty()) {
            layer_cand.push_back(topRes.top());
            topRes.pop();
        }
        std::sort(layer_cand.begin(), layer_cand.end(),
                  [](const Pair& a, const Pair& b) { return a.first < b.first; });

        // 选出 maxM 个邻居
        std::vector<int> selected;
        int maxM = maxMForLevel(l);
        select_neighbors_heuristic(layer_cand, maxM, selected);

        // ★新增：过滤 selected（防止脏边污染图，避免 reverse-edge 后处理中越界崩溃）
        selected.erase(
            std::remove_if(selected.begin(), selected.end(),
                [&](int x){ return x < 0 || x >= N_ || x == cur; }),
            selected.end());

        // 把当前点的邻居写入 graph_[l][cur]
#if ABLATE_SERIAL_BUILD
        {
            auto& neigh = graph_[l][cur];
            neigh.insert(neigh.end(), selected.begin(), selected.end());

            // cur 自己也要去重 + 超限剪枝（你已经加了，保留）
            if (!neigh.empty()) {
                std::sort(neigh.begin(), neigh.end());
                neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());
            }

            int maxM2 = maxMForLevel(l);
            if ((int)neigh.size() > maxM2) {
                std::vector<Pair> cand_cur;
                cand_cur.reserve(neigh.size());
                const VecType* curV = getVec(cur);
                for (int other : neigh) {
                    if (other < 0 || other >= N_ || other == cur) continue; // 再保险
                    float d = dist_vec(getVec(other), curV);
                    cand_cur.emplace_back(d, other);
                }
                std::vector<int> pruned;
                select_neighbors_heuristic(cand_cur, maxM2, pruned);
                neigh.assign(pruned.begin(), pruned.end());

                // 再过滤一次
                neigh.erase(std::remove_if(neigh.begin(), neigh.end(),
                    [&](int x){ return x < 0 || x >= N_ || x == cur; }),
                    neigh.end());
            }
        }
#else
        {
            std::lock_guard<std::mutex> lk(*nodeLocks_[cur]);
            auto& neigh = graph_[l][cur];
            neigh.insert(neigh.end(), selected.begin(), selected.end());

            // cur 自己也要去重 + 超限剪枝（你已经加了，保留）
            if (!neigh.empty()) {
                std::sort(neigh.begin(), neigh.end());
                neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());
            }

            int maxM2 = maxMForLevel(l);
            if ((int)neigh.size() > maxM2) {
                std::vector<Pair> cand_cur;
                cand_cur.reserve(neigh.size());
                const VecType* curV = getVec(cur);
                for (int other : neigh) {
                    if (other < 0 || other >= N_ || other == cur) continue; // 再保险
                    float d = dist_vec(getVec(other), curV);
                    cand_cur.emplace_back(d, other);
                }
                std::vector<int> pruned;
                select_neighbors_heuristic(cand_cur, maxM2, pruned);
                neigh.assign(pruned.begin(), pruned.end());

                // 再过滤一次
                neigh.erase(std::remove_if(neigh.begin(), neigh.end(),
                    [&](int x){ return x < 0 || x >= N_ || x == cur; }),
                    neigh.end());
            }
        }
#endif

        // 对每个邻居做双向连接，并且如果邻居度超过限制就剪枝
        for (int nb : selected) {
            if (nb < 0 || nb >= N_ || nb == cur) continue; // 保险
#if ABLATE_SERIAL_BUILD
            {
                auto& neigh_nb = graph_[l][nb];
                neigh_nb.push_back(cur);

                if ((int)neigh_nb.size() > maxMForLevel(l)) {
                    // 对邻居 nb 的邻接表做一次剪枝
                    std::vector<Pair> cand_nb;
                    cand_nb.reserve(neigh_nb.size());
                    const VecType* nbVec = getVec(nb);
                    for (int other : neigh_nb) {
                        if (other < 0 || other >= N_ || other == nb) continue;
                        float d = dist_vec(getVec(other), nbVec);
                        cand_nb.emplace_back(d, other);
                    }
                    std::vector<int> pruned;
                    select_neighbors_heuristic(cand_nb, maxMForLevel(l), pruned);
                    neigh_nb.assign(pruned.begin(), pruned.end());

                    // 去掉 self/非法（保险）
                    neigh_nb.erase(std::remove_if(neigh_nb.begin(), neigh_nb.end(),
                        [&](int x){ return x < 0 || x >= N_ || x == nb; }),
                        neigh_nb.end());
                }
            }
#else
            {
                std::lock_guard<std::mutex> lk(*nodeLocks_[nb]);
                auto& neigh_nb = graph_[l][nb];
                neigh_nb.push_back(cur);

                if ((int)neigh_nb.size() > maxMForLevel(l)) {
                    // 对邻居 nb 的邻接表做一次剪枝
                    std::vector<Pair> cand_nb;
                    cand_nb.reserve(neigh_nb.size());
                    const VecType* nbVec = getVec(nb);
                    for (int other : neigh_nb) {
                        if (other < 0 || other >= N_ || other == nb) continue;
                        float d = dist_vec(getVec(other), nbVec);
                        cand_nb.emplace_back(d, other);
                    }
                    std::vector<int> pruned;
                    select_neighbors_heuristic(cand_nb, maxMForLevel(l), pruned);
                    neigh_nb.assign(pruned.begin(), pruned.end());

                    // 去掉 self/非法（保险）
                    neigh_nb.erase(std::remove_if(neigh_nb.begin(), neigh_nb.end(),
                        [&](int x){ return x < 0 || x >= N_ || x == nb; }),
                        neigh_nb.end());
                }
            }
#endif
        }

        // 下一层的起始点设为当前层的一个最近邻（如果有）
        if (!selected.empty()) {
            ep = selected[0];
        }
    }
}


// ======================= Search API =======================

void Solution::search(const std::vector<float>& q, int* res) {
    if (N_ == 0) {
        for (int i = 0; i < 10; ++i) res[i] = 0;
        return;
    }

    // 1. 量化查询向量
#if ABLATE_USE_FLOAT_DISTANCE
    const VecType* qPtr = q.data();
#else
    std::vector<int16_t> q_sq(dim_aligned_);
    quantize_vec(q.data(), q_sq.data());
    const VecType* qPtr = q_sq.data();
#endif

    // 2. 从最高层贪心到第 0 层
    int ep;
    int curMax;
    {
        std::lock_guard<std::mutex> g(globalLock_);
        ep = enterPoint_;
        curMax = maxLevel_;
    }
    for (int l = curMax; l > 0; --l) {
        ep = greedy_search_layer(qPtr, ep, l);
    }

    // 3. 在第 0 层用 beam search 找到 efS_ 个候选
    const int ef = (efS_ > 0 ? efS_ : 10);
    std::vector<Pair> cand = search_layer_with_dist(qPtr, ep, ef, 0);

    // 4. 写回前 10 个结果
    for (int i = 0; i < 10; ++i) {
        res[i] = (i < (int)cand.size()) ? cand[i].second : 0;
    }
}
