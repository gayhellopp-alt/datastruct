# CV 项目描述

以下措辞依据项目源码和课程报告整理。项目时间可写为 **2025.11–2026.01**；课程报告日期为 2026 年 1 月 11 日。GitHub 链接为 <https://github.com/gayhellopp-alt/datastruct>。

## English

**HNSW Approximate Nearest Neighbor Search | C++, AVX2, Multithreading**  
Fudan University, Data Structures H course project | Nov 2025 – Jan 2026

- Implemented an HNSW-based top-10 vector search index with LID-guided construction ordering, level-0 graph refinement, and gamma-based query stopping.
- Combined SQ16 quantization, AVX2 distance computation, flattened adjacency storage, CPU prefetching, and parallel graph construction; conducted ablation studies of recall, query latency, build time, and distance computation counts.

Optional quantitative bullet, if the application benefits from a reported evaluation result:

- Recorded 98.91% Recall@10 at 1.0577 ms/query on the course grading server, as documented in the final project report.

For a compact one-line project entry:

Implemented and evaluated an HNSW-based ANN index in C++, combining LID-guided construction, query stopping, SQ16/AVX2, and parallel graph construction.

## 中文

**基于 HNSW 的近似最近邻检索 | C++、AVX2、多线程**  
复旦大学《数据结构 H》课程项目 | 2025.11–2026.01

- 实现面向高维向量 Top-10 检索的 HNSW 图索引，结合 LID 引导的插入顺序重排、底层图后处理和 gamma 查询早停策略。
- 结合 SQ16 量化、AVX2 距离计算、邻接表扁平化、CPU 预取和并行建图，并通过消融实验分析召回率、查询时延、构建时间及距离计算次数之间的权衡。

需要量化结果时可补充：

- 最终课程报告记录测评机 Recall@10 为 98.91%，平均查询时延为 1.0577 ms。

精简版：

使用 C++ 实现并分析 HNSW 向量检索索引，探索 LID 构建顺序、查询早停与 SQ16/AVX2、并行建图等优化。

## 使用说明

优先使用前两条技术描述，能直接对应源码与实验分析。若保留数值，保持测试环境和指标对应关系；不要将本地 99.6% 与测评机 1.0577 ms 合并。现有材料未提供排名、获奖、课程成绩或线上部署记录，因此上述文案不包含这些表述。
