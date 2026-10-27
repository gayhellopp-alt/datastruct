# 项目经历

[GitHub](https://github.com/gayhellopp-alt/datastruct) · [课程报告](report.pdf)

## English

**HNSW Approximate Nearest Neighbor Search | C++, AVX2, Multithreading**  
Fudan University, Data Structures H course project | Nov 2025 – Jan 2026

- Implemented an HNSW-based top-10 vector search index with LID-guided construction ordering, level-0 graph refinement, and gamma-based query stopping.
- Combined SQ16 quantization, AVX2 distance computation, flattened adjacency storage, CPU prefetching, and parallel graph construction; evaluated their effects through ablation experiments.
- Achieved 98.91% Recall@10 with an average query latency of 1.0577 ms on the course grading server.

## 中文

**基于 HNSW 的近似最近邻检索 | C++、AVX2、多线程**  
复旦大学《数据结构 H》课程项目 | 2025.11–2026.01

- 实现面向高维向量 Top-10 检索的 HNSW 图索引，结合 LID 引导的插入顺序重排、底层图后处理和 gamma 查询早停策略。
- 结合 SQ16 量化、AVX2 距离计算、邻接表扁平化、CPU 预取和并行建图，通过消融实验比较各模块对召回率、查询时延、构建时间及距离计算次数的影响。
- 在课程测评机上达到 98.91% 的 Recall@10，平均查询时延为 1.0577 ms。
