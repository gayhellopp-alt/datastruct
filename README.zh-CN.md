# 基于 HNSW 的近似最近邻检索

**通过构建顺序优化与查询早停，探索高维向量检索的精度、时延和构建成本之间的权衡。**

[English](README.md) · [课程报告](docs/report.pdf) · [算法说明](docs/method.md) · [实验结果](docs/results.md)

作者：**张弛 / Zhang Chi**。本项目为复旦大学 **2025–2026 学年秋季学期《数据结构 H》** 课程 PJ，报告完成于 **2026 年 1 月 11 日**。

本次整理将课程最终提交、报告和实验记录组织成便于阅读的项目档案。**`src/` 中两份源文件与提供的原始文件逐字节一致；此次未修改算法、未重新运行性能测评。**

## 项目工作

- 实现 HNSW 风格的分层图索引、上层贪心导航、底层候选搜索与邻居多样性筛选。
- 使用两遍建图流程：先粗建图估计局部内禀维度（LID），再按估计难度重排插入顺序。
- 在第 0 层执行反向边注入和度数约束剪枝。
- 使用 gamma 早停条件和候选入队阈值控制查询扩展。
- 结合 SQ16、AVX2、扁平化邻接表、CPU 预取、并行建图和 visited 版本号优化。
- 通过消融实验分析构建顺序、图后处理、查询策略及实现优化的影响。

## 原报告记录的结果

| 测试环境 | Recall@10 | 平均查询时延 | 平均距离计算次数 |
| --- | ---: | ---: | ---: |
| 最终提交，本地，更新前 GloVe 数据集 | 99.6% | 0.718538 ms | 约 12,970 |
| 相同代码，本地，更新后 GloVe 数据集 | 99.6% | 0.800284 ms | 14,362.490 |
| 课程测评机，报告记录的提交结果 | 98.91% | 1.0577 ms | 未提供 |

以上转录自报告末尾说明，属于**当时作者记录的数据**，与报告中的消融实验 baseline 分开呈现。当前未提供完整数据集、测评机环境和逐次运行日志，不能将不同环境或不同数据版本的指标混为同一次实验。完整表格和来源见 [实验说明](docs/results.md)。

## 文件导航

| 路径 | 内容 |
| --- | --- |
| [`src/`](src/) | 最终提交代码，保持原样 |
| [`docs/report.pdf`](docs/report.pdf) | 中文报告正文；公开版替换封面，去除学号 |
| [`docs/course-context.md`](docs/course-context.md) | 课程目标、接口、数据集和评价方式摘要 |
| [`docs/method.md`](docs/method.md) | 算法流程和提交参数 |
| [`docs/results.md`](docs/results.md) | 消融实验、最终结果与解释边界 |
| [`results/`](results/) | 从报告整理出的 CSV 数据 |
| [`docs/cv.md`](docs/cv.md) | 可用于中英文 CV 的项目描述 |
| [`docs/archive-notes.md`](docs/archive-notes.md) | 原文件校验值、整理记录、旧版文件说明 |

根目录原有 `MySolution_after.*`、`main.cpp`、sweep 占位文件和 NGT 子模块均保留。它们是较早版本的历史材料；与本次报告一并提供的最终源码以 **`src/MySolution.cpp` 和 `src/MySolution.h`** 为准。

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

该命令生成目标文件；完整运行需要课程评测入口或兼容驱动和数据集。根目录 `main.cpp` 是原有本地测试驱动，使用固定的数据路径和底库前缀查询，不应视为独立查询集上的评估，也不保证复现课程测评机结果。更多说明见 [整理记录](docs/archive-notes.md)。
