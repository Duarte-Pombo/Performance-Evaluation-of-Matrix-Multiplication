# Performance Evaluation of Matrix Multiplication

## Project Overview
This repository contains the implementation and analysis of various matrix multiplication algorithms to study the effect of memory hierarchy and multi-core execution on processor performance. The project is divided into two primary parts: single-core performance evaluation (comparing C++ and Rust) and multi-core performance evaluation (using OpenMP in C++).

The computational complexity for all implemented matrix multiplication algorithms is 2n^3 FLOPs for matrices of size n×n.

## Single-Core Performance

This section analyzes the impact of memory access patterns and cache utilization on a single CPU core.
Implemented Algorithms

1. Basic Algorithm: Multiplies one row of the first matrix by each column of the second matrix. Implemented in C++ and Rust.

2. Line-by-Line Algorithm: Multiplies an element from the first matrix by the corresponding row of the second matrix, significantly improving spatial locality. Implemented in C++ and Rust.

3. Block-Oriented Algorithm: Divides the matrices into sub-blocks (e.g., 128, 256, 512) to fit data into the CPU cache, minimizing cache misses. Implemented in C++ only.

### Metrics and Tooling

- Execution Time: Measured across varying matrix dimensions (from 1024×1024 up to 10240×10240).

- Hardware Counters: Linux perf is used to measure hardware metrics, specifically focusing on cache efficiency (mem_load_retired.l1_miss, mem_load_retired.l2_miss).

## Part 2: Multi-Core Performance

This section focuses on parallelizing the matrix multiplication algorithms using OpenMP to leverage multi-core processors.

### Parallel Implementations

1. Basic and Line-by-Line Parallelization: Analyzing varying OpenMP pragma placements (e.g., inner loop vs. outer loop parallelization) using 4 threads on matrix sizes ranging from 1024 to 3072.

2. Advanced Directives: Applying optimizations to the Line-by-Line algorithm for an 8192×8192 matrix using 4 to 24 threads. This includes exploring:
 - `pragma omp for simd` for vectorization.
 - `pragma omp parallel for collapse(2)` for expanding the parallel iteration space.

### Evaluated Metrics

- Throughput: GFlop/s.

- Scalability: Speedup and Efficiency relative to the single-core baseline.

Repository Structure

- `src/cpp/`: C++ source files for basic, line-by-line, blocked, and OpenMP versions.

- `src/rust/`: Rust source files for basic and line-by-line single-core versions.

- `scripts/`: Bash scripts for automating compilation, execution, and perf data collection.

- `results/`: Output logs, CSV files with execution times, and raw perf output.

- `analysis/`: Generated graphs plotting GFlop/s, Speedup, and Efficiency, alongside the final analytical report.

## Prerequisites

To compile and run the programs, the following tools are required:

- C++ Compiler: GCC with C++11 (or higher) and OpenMP support.

- Rust Toolchain: `rustc` and `cargo`.

- Profiling Tool: Linux `perf` (Ensure `kernel.perf_event_paranoid` is set to allow hardware counter profiling).

## Compilation and Execution

### C++ Implementations

Navigate to the C++ source directory and compile using the `-O2` optimization flag and OpenMP.

### Rust Implementations

Navigate to the Rust project directory and compile using the release profile for maximum optimization.
Profiling with Perf

Use the provided automation scripts or run `perf` directly. Example command to measure L1 and L2 cache misses:
Results and Analysis

