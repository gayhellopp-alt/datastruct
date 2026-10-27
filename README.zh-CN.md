# 基于 HNSW 的近似最近邻检索

**通过构建顺序优化与查询早停，探索高维向量检索的精度、时延和构建成本之间的权衡。**

[English](README.md) · [课程报告](docs/report.pdf) · [算法说明](docs/method.md) · [实验结果](docs/results.md)

这是我在复旦大学 **2025–2026 学年秋季学期《数据结构 H》** 课程中完成的 PJ。我以 HNSW 为基础，研究构建顺序、底层图结构和查询扩展策略对 Top-10 向量检索的影响。

我的目标是在保持较高召回率的同时降低查询时延，并通过消融实验理解各项优化的作用。仓库包含我的最终 C++ 实现、实验结果和 [课程报告](docs/report.pdf)。报告完成于 **2026 年 1 月 11 日**。

## 我做了什么

- 实现 HNSW 风格的分层图索引、上层贪心导航、底层候选搜索与邻居多样性筛选。
- 使用两遍建图流程：先粗建图估计局部内禀维度（LID），再按估计难度重排插入顺序。
- 在第 0 层执行反向边注入和度数约束剪枝。
- 使用 gamma 早停条件和候选入队阈值控制查询扩展。
- 结合 SQ16、AVX2、扁平化邻接表、CPU 预取、并行建图和 visited 版本号优化。
- 通过消融实验分析构建顺序、图后处理、查询策略及实现优化的影响。

## 实验结果

我用最终版本在本地测试了更新前、后的 GloVe 数据集，并提交到课程测评机：

| 测试环境 | Recall@10 | 平均查询时延 | 平均距离计算次数 |
| --- | ---: | ---: | ---: |
| 本地，更新前 GloVe 数据集 | 99.6% | 0.718538 ms | 约 12,970 |
| 本地，更新后 GloVe 数据集 | 99.6% | 0.800284 ms | 14,362.490 |
| 课程测评机 | 98.91% | 1.0577 ms | — |

此外，我分别对 LID 重排、图后处理、查询早停、量化距离、并行建图和预取做了对照实验。完整表格与分析见 [实验结果](docs/results.md)。

## 文件导航

| 路径 | 内容 |
| --- | --- |
| [`src/`](src/) | 最终提交代码 |
| [`docs/report.pdf`](docs/report.pdf) | 中文课程报告 |
| [`docs/course-context.md`](docs/course-context.md) | 课程目标、接口、数据集和评价方式 |
| [`docs/method.md`](docs/method.md) | 算法流程和参数设置 |
| [`docs/results.md`](docs/results.md) | 消融实验与最终结果 |
| [`results/`](results/) | CSV 格式的实验数据表 |
| [`docs/cv.md`](docs/cv.md) | 中英文简历项目描述 |
| [`docs/archive-notes.md`](docs/archive-notes.md) | 项目文件与早期版本说明 |

我把最终提交放在 **`src/`**，较早的实现和本地测试工具保留在仓库根目录。

## 使用接口

```cpp
Solution index;
index.build(dimension, base_flat); // 按行展开的 N × dimension 浮点数组
int ids[10];
index.search(query, ids);         // 返回 10 个从 0 开始的向量 ID
```

代码使用 C++14 特性和 x86 指令。支持 AVX2 的 x86 环境下，GCC 编译示例：

```sh
mkdir -p build
g++ -std=c++14 -O3 -mavx2 -pthread -Isrc -c src/MySolution.cpp -o build/MySolution.o
```

生成的目标文件可与课程评测入口或兼容驱动链接。不加 `-mavx2` 时，代码使用标量距离分支。

我早期的本地测试入口在 `main.cpp`，读取 `./data_o/glove/base.txt`，取底库前 100 个向量作为查询。文件对应关系见 [项目文件说明](docs/archive-notes.md)。
