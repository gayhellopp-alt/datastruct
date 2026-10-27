# HNSW Approximate Nearest Neighbor Search

**Construction ordering and query early stopping for high-dimensional vector retrieval.**

[中文说明](README.zh-CN.md) · [Project report](docs/report.pdf) · [Method](docs/method.md) · [Experiments](docs/results.md)

A C++ course project by **Zhang Chi** for **Data Structures H**, Fudan University, Fall 2025. The project implements a hierarchical graph index for top-10 approximate nearest neighbor (ANN) search and studies the trade-off between recall, query latency, and index construction cost.

The final report was completed on **January 11, 2026**. This repository organizes the original submission and its supporting documentation for a project portfolio. **The source files in `src/` are byte-for-byte copies of the submitted files; this archival update makes no algorithm changes and reports no new benchmark runs.**

## What I implemented

- **HNSW-based indexing:** hierarchical navigation, greedy descent, beam-style search, and diversity-based neighbor selection.
- **LID-guided construction ordering:** a coarse first-pass graph estimates local difficulty, followed by a second build with nodes ordered by estimated local intrinsic dimensionality.
- **Level-0 graph refinement:** reverse-edge injection and degree-constrained pruning.
- **Query control:** a gamma-based stopping condition and a distance threshold for admitting candidates to the search queue.
- **Systems optimizations:** SQ16 vector storage, AVX2 distance computation, flattened level-0 adjacency, CPU prefetching, parallel construction, and versioned visited arrays.
- **Experimental analysis:** ablations of ordering, graph post-processing, query control, distance representation, parallelism, and prefetching.

## Results recorded in the report

The final report distinguishes local measurements on two dataset versions from a result on the course grading server:

| Evaluation setting | Recall@10 | Average query latency | Distance computations/query |
| --- | ---: | ---: | ---: |
| Final submission, local, original GloVe dataset | 99.6% | 0.718538 ms | approximately 12,970 |
| Same code, local, updated GloVe dataset | 99.6% | 0.800284 ms | 14,362.490 |
| Course grading server, reported submission | 98.91% | 1.0577 ms | not reported |

These are **historical, author-reported measurements**, transcribed from the report's final note. They are separate from the report's ablation baseline. Dataset files, complete run logs, and the grading environment are not bundled. See [experiment provenance and comparisons](docs/results.md) before interpreting the numbers.

## Repository guide

| Location | Contents |
| --- | --- |
| [`src/`](src/) | Original final submission: `MySolution.cpp` and `MySolution.h` |
| [`docs/report.pdf`](docs/report.pdf) | Original Chinese report body with a public cover omitting the student ID |
| [`docs/course-context.md`](docs/course-context.md) | Assignment objective, required interface, datasets, and metrics |
| [`docs/method.md`](docs/method.md) | Implementation overview and submitted parameter values |
| [`docs/results.md`](docs/results.md) | Ablation tables, final results, and measurement context |
| [`results/`](results/) | CSV transcriptions of the report's numerical tables |
| [`docs/cv.md`](docs/cv.md) | English and Chinese project descriptions for a CV |
| [`docs/archive-notes.md`](docs/archive-notes.md) | File provenance, preservation policy, and legacy file guide |

The root-level `MySolution_after.*`, `main.cpp`, sweep placeholders, and NGT submodule predate this archive. They are retained as historical material. Use **`src/MySolution.cpp` with `src/MySolution.h`** when referring to the final submission supplied with the report.

## Interface and compilation

```cpp
Solution index;
index.build(dimension, base_flat); // N * dimension floats, row-major
int ids[10];
index.search(query, ids);         // 10 zero-based vector IDs
```

The implementation uses C++14 features and x86 intrinsics. An example compilation command for GCC on an x86 machine with AVX2 support is:

```sh
mkdir -p build
g++ -std=c++14 -O3 -mavx2 -pthread -Isrc -c src/MySolution.cpp -o build/MySolution.o
```

This produces an object file. The course evaluator or a compatible driver must provide the executable entry point and input data. Compiling without `-mavx2` selects the source's scalar distance branch; the file still includes x86 prefetch intrinsics.

The existing `main.cpp` is an older local benchmark driver with a hard-coded GloVe path and queries taken from the base-vector prefix. Its query protocol should not be treated as a held-out evaluation or assumed to reproduce the grading server. [Archive notes](docs/archive-notes.md) describe how it differs from the final source snapshot.

## Project context

The assignment concerns efficient vector retrieval, a core operation used in embedding search. This project focuses on algorithm implementation and empirical optimization within that coursework setting. It does not implement an end-to-end RAG application or a vector database service.
