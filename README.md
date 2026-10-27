# HNSW Approximate Nearest Neighbor Search

**Construction ordering and query early stopping for high-dimensional vector retrieval.**

[中文说明](README.zh-CN.md) · [Project report](docs/report.pdf) · [Method](docs/method.md) · [Experiments](docs/results.md)

I built this project for **Data Structures H** at **Fudan University** in Fall 2025. Starting from HNSW, I explored how construction order, graph refinement, and query stopping affect top-10 approximate nearest neighbor (ANN) search.

My goal was to reduce query latency while maintaining high recall, then understand the cost of each optimization through ablation experiments. The repository contains my final C++ implementation, experimental results, and [course report](docs/report.pdf), completed on **January 11, 2026**.

## What I implemented

- **HNSW-based indexing:** hierarchical navigation, greedy descent, beam-style search, and diversity-based neighbor selection.
- **LID-guided construction ordering:** a coarse first-pass graph estimates local difficulty, followed by a second build with nodes ordered by estimated local intrinsic dimensionality.
- **Level-0 graph refinement:** reverse-edge injection and degree-constrained pruning.
- **Query control:** a gamma-based stopping condition and a distance threshold for admitting candidates to the search queue.
- **Systems optimizations:** SQ16 vector storage, AVX2 distance computation, flattened level-0 adjacency, CPU prefetching, parallel construction, and versioned visited arrays.
- **Experimental analysis:** ablations of ordering, graph post-processing, query control, distance representation, parallelism, and prefetching.

## Results

I tested the final implementation locally on the original and updated GloVe datasets, and submitted it to the course grading server:

| Evaluation setting | Recall@10 | Average query latency | Distance computations/query |
| --- | ---: | ---: | ---: |
| Local, original GloVe dataset | 99.6% | 0.718538 ms | approximately 12,970 |
| Local, updated GloVe dataset | 99.6% | 0.800284 ms | 14,362.490 |
| Course grading server | 98.91% | 1.0577 ms | — |

I also compared LID ordering, graph post-processing, query stopping, quantization, parallel construction, and prefetching. The full tables and my analysis are in [Experiments](docs/results.md).

## Repository guide

| Location | Contents |
| --- | --- |
| [`src/`](src/) | Final submission: `MySolution.cpp` and `MySolution.h` |
| [`docs/report.pdf`](docs/report.pdf) | Course report in Chinese |
| [`docs/course-context.md`](docs/course-context.md) | Assignment objective, required interface, datasets, and metrics |
| [`docs/method.md`](docs/method.md) | Implementation overview and parameter values |
| [`docs/results.md`](docs/results.md) | Ablation experiments and final results |
| [`results/`](results/) | Experiment tables in CSV format |
| [`docs/cv.md`](docs/cv.md) | English and Chinese project descriptions for a CV |
| [`docs/archive-notes.md`](docs/archive-notes.md) | Notes on the project files and earlier versions |

I keep earlier implementation files and local tools at the repository root. My final submission is in **`src/`**.

## Interface and compilation

```cpp
Solution index;
index.build(dimension, base_flat); // N * dimension floats, row-major
int ids[10];
index.search(query, ids);         // 10 zero-based vector IDs
```

The implementation uses C++14 features and x86 intrinsics. To compile it with GCC on an x86 machine with AVX2 support:

```sh
mkdir -p build
g++ -std=c++14 -O3 -mavx2 -pthread -Isrc -c src/MySolution.cpp -o build/MySolution.o
```

The resulting object file can be linked with the course evaluator or a compatible driver. Removing `-mavx2` selects the scalar distance branch.

My earlier local benchmark is in `main.cpp`. It reads `./data_o/glove/base.txt` and uses the first 100 base vectors as queries. See [Project files](docs/archive-notes.md) for the file layout.
