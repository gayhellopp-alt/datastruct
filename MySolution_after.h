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

private:
    using IdType   = int;
    using DistType = float;
    using Pair     = std::pair<DistType, IdType>;

    int dim_ = 0;
    int N_   = 0;

    // SQ16 量化相关参数
    int   dim_aligned_  = 0;      // 对齐到 16 的维度
    float min_val_      = 0.0f;   // 全局最小值
    float diff_scale_   = 0.0f;   // 量化缩放因子
    std::vector<int16_t> data_sq_; // 量化后的数据 N * dim_aligned_

    // 图结构参数
    int maxLevel_   = -1;
    int enterPoint_ = -1;
    int M_   = 16;    // 上层最大度
    int M0_  = 256;    // 底层最大度
    int efC_ = 256;   // 构建时 efConstruction
    int efS_ = 10;   // 搜索时 efSearch

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

    std::mt19937 rng_;

    // 每个节点一把锁，构建时保护 graph_ 的邻接表
    std::vector<std::unique_ptr<std::mutex>> nodeLocks_;
    mutable std::mutex globalLock_; // 保护 enterPoint_/maxLevel_ 的更新

    // ===== 量化 & 距离计算 =====
    void quantize_vec(const float* src, int16_t* dst) const;
    float dist_sq16(const int16_t* d1, const int16_t* d2) const;
    DistType dist2_float(const float* a, const float* b) const; // 备用（目前基本不用）

    inline const int16_t* getVecSQ(IdType id) const {
        return &data_sq_[(std::size_t)id * dim_aligned_];
    }

    // ===== HNSW 结构相关 =====
    int sampleLevel();
    inline int maxMForLevel(int level) const { return (level == 0) ? M0_ : M_; }

    // 上层贪心搜索（使用 graph_[level]）
    int greedy_search_layer(const int16_t* q_sq, int ep, int level) const;

    // 底层 beam search（搜索接口用；level=0 时会走扁平化图）
    std::vector<Pair> search_layer_with_dist(const int16_t* q_sq,
                                             int ep,
                                             int ef,
                                             int level) const;

    // 构建时的候选选择（RNG / heuristic）
    void select_neighbors_heuristic(std::vector<Pair>& candidates,
                                    int M,
                                    std::vector<int>& out) const;

    // ===== 并行构建 =====
    void addPointMT(int cur,
                    const int16_t* curVecSQ,
                    std::vector<std::uint32_t>& visited_local,
                    std::uint32_t& visitedTokenLocal);

    void search_layer_mt(const int16_t* q_sq,
                         int ep,
                         int ef,
                         int level,
                         std::priority_queue<Pair>& topRes,
                         std::vector<std::uint32_t>& visited_local,
                         std::uint32_t& visitedTokenLocal);
};

#endif // MYSOLUTION_H
