# 项目文件说明

我把最终提交放在 `src/`，课程报告和实验分析放在 `docs/`，较早的实现及本地测试工具保留在根目录。

| 文件 | 用途 |
| --- | --- |
| `src/MySolution.cpp` / `src/MySolution.h` | 最终提交，包含 LID 两遍建图、底层图后处理和 gamma 查询早停 |
| `docs/report.pdf` | 中文课程报告，公开版封面省去学号 |
| `results/reported-ablation.csv` | 消融实验数据 |
| `results/reported-final.csv` | 最终版本在本地和测评机上的结果 |
| `MySolution_after.cpp` / `MySolution_after.h` | 较早的实现版本 |
| `main.cpp` | 早期本地测试入口，读取 `./data_o/glove/base.txt`，取底库前 100 个向量作为查询 |
| `sweep.py` / `sweep2.py` | 空白占位文件 |
| `.gitmodules` / `third_party/NGT` | NGT 子模块引用 |

课程原课件和带原封面的报告保留在本地材料中。仓库里的 [课程项目](course-context.md)、[方法说明](method.md) 和 [实验结果](results.md) 分别介绍题目、实现和测试过程。
