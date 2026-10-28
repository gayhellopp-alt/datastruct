#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <random>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include <memory>
#include <cmath>

// Ablation macros (set to 1 to enable)
#ifndef ABLATE_DISABLE_GAMMA_EARLY_STOP
#define ABLATE_DISABLE_GAMMA_EARLY_STOP 0
#endif
#ifndef ABLATE_CAND_ALWAYS_ENQUEUE
#define ABLATE_CAND_ALWAYS_ENQUEUE 0
#endif
#ifndef ABLATE_USE_FLOAT_DISTANCE
#define ABLATE_USE_FLOAT_DISTANCE 0
#endif
#ifndef ABLATE_SERIAL_BUILD
#define ABLATE_SERIAL_BUILD 0
#endif
#ifndef ABLATE_DISABLE_POST_PROCESS
#define ABLATE_DISABLE_POST_PROCESS 0
#endif
#ifndef ABLATE_DISABLE_PREFETCH
#define ABLATE_DISABLE_PREFETCH 0
#endif

// 保持对齐，方便 SIMD 加载
#ifdef _MSC_VER
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif

class Solution {
public:
    Solution();
    // dim: 向量维度；base_flat: N * dim 的扁平数组
    void build(int dim, const std::vector<float>& base_flat);
    // q: 查询向量；res: 输出前 10 个近邻 id（main.cpp 的 TOPK=10）
    void search(const std::vector<float>& q, int* res);

    //测量使用
    static void reset_dist_counter();
    static std::uint64_t get_dist_counter();
    static void enable_dist_counter(bool on);

private:
    using IdType   = int;
    using DistType = float;
    using Pair     = std::pair<DistType, IdType>;
#if ABLATE_USE_FLOAT_DISTANCE
    using VecType  = float;
#else
    using VecType  = int16_t;
#endif

    int dim_ = 0;
    int N_   = 0;

    // SQ16 量化相关参数
    int   dim_aligned_  = 0;      // 对齐到 16 的维度
    float min_val_      = 0.0f;   // 全局最小值
    float diff_scale_   = 0.0f;   // 量化缩放因子
    std::vector<int16_t> data_sq_; // 量化后的数据 N * dim_aligned_
    std::vector<float> data_float_; // float 数据 N * dim

    // 图结构参数
    int maxLevel_   = -1;
    int enterPoint_ = -1;
    int M_   = 16;    // 上层最大度
    int M0_  = 54;    // 底层最大度
    int efC_ = 500;   // 构建时 efConstruction
    int efS_ = 10;   // 搜索时 efSearch
    float gamma_ = 0.21f;

    // 原始图结构：graph_[level][node] = 邻接表
    std::vector<std::vector<std::vector<int>>> graph_;

    // 第 0 层扁平化图 (搜索时使用)
    // 布局：[deg, nb0, nb1, ..., pad, deg, ...]，stride = M0_ + 1
    std::vector<int> flat_graph_l0_;

    // 每个节点所在的最高层
    std::vector<int> nodeLevel_;

    // 全局 visited（仅搜索阶段用），版本号技巧避免每次清零
    mutable std::vector<std::uint32_t> visited_;
    mutable std::uint32_t visitedToken_ = 1u;

    //测量使用
    static std::atomic<std::uint64_t> dist_counter_;
    static std::atomic<bool> dist_count_enabled_;

    std::mt19937 rng_;

    // 每个节点一把锁，构建时保护 graph_ 的邻接表
    std::vector<std::unique_ptr<std::mutex>> nodeLocks_;
    mutable std::mutex globalLock_; // 保护 enterPoint_/maxLevel_ 的更新

    // ===== 量化 & 距离计算 =====
    void quantize_vec(const float* src, int16_t* dst) const;
    float dist_sq16(const int16_t* d1, const int16_t* d2) const;
    DistType dist2_float(const float* a, const float* b) const; // 备用（目前基本不用）

    inline const VecType* getVec(IdType id) const {
#if ABLATE_USE_FLOAT_DISTANCE
        return &data_float_[(std::size_t)id * dim_];
#else
        return &data_sq_[(std::size_t)id * dim_aligned_];
#endif
    }

    inline DistType dist_vec(const VecType* a, const VecType* b) const {
#if ABLATE_USE_FLOAT_DISTANCE
        return dist2_float(a, b);
#else
        return dist_sq16(a, b);
#endif
    }

    // ===== HNSW 结构相关 =====
    int sampleLevel();
    inline int maxMForLevel(int level) const { return (level == 0) ? M0_ : M_; }

    // 上层贪心搜索（使用 graph_[level]）
    int greedy_search_layer(const VecType* q_vec, int ep, int level) const;

    // 底层 beam search（搜索接口用；level=0 时会走扁平化图）
    std::vector<Pair> search_layer_with_dist(const VecType* q_vec,
                                             int ep,
                                             int ef,
                                             int level) const;

    // 构建时的候选选择（RNG / heuristic）
    void select_neighbors_heuristic(std::vector<Pair>& candidates,
                                    int M,
                                    std::vector<int>& out) const;

    // ===== 并行构建 =====
    void addPointMT(int cur,
                    const VecType* curVec,
                    std::vector<std::uint32_t>& visited_local,
                    std::uint32_t& visitedTokenLocal);

    void search_layer_mt(const VecType* q_vec,
                         int ep,
                         int ef,
                         int level,
                         std::priority_queue<Pair>& topRes,
                         std::vector<std::uint32_t>& visited_local,
                         std::uint32_t& visitedTokenLocal);


    
};

#endif // MYSOLUTION_H
