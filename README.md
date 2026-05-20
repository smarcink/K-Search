<p align="center">
  <h1 align="center">🔍 K-Search</h1>
  <p align="center">
    <b>LLM-Driven GPU Kernel Optimization with Co-Evolving Intrinsic World Model</b>
  </p>
  <p align="center">
    <i>Automatically generate, evaluate, and iteratively optimize high-performance GPU kernels using frontier LLMs guided by a co-evolving world model.</i>
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2602.19128v1"><img src="https://img.shields.io/badge/arXiv-2602.19128-b31b1b.svg" alt="arXiv"></a>
    <a href="https://github.com/caoshiyi/K-Search/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  </p>
</p>

<p align="center">
  <img src="assets/overview-2.png" alt="K-Search System Overview" width="90%"/>
</p>

---

## Overview

**K-Search** is an automated kernel engineering system that uses large language models (GPT-5, Gemini etc.) to iteratively generate and optimize GPU kernels. Unlike one-shot code generation, K-Search maintains a **co-evolving world model** — a structured search tree that encodes hypotheses about kernel bottlenecks, design alternatives, and optimization strategies — guiding multi-round, evidence-driven search over the kernel design space efficiently.

## Features

- ⚡ **Task Backends** — Two focused backends for kernel optimization:
  - **cuda_kernel** — Multi-file CUDA kernel optimization (kernel.h / kernel.cu / main.cpp)
  - **device_bench** — Triton kernel optimization on NVIDIA CUDA and Intel XPU

- 📊 **W&B Integration** — Full Weights & Biases logging with per-round score tracking, generated code artifacts, and world model snapshots.

- 🔌 **Multi-Model Support** — Works with any OpenAI-compatible API endpoint:
  - OpenAI: `gpt-5.2`, `o3`
  - Google: `gemini-3-pro-preview` (via compatible endpoint)
  - Any model exposing a chat completions API

- 💾 **Solution Persistence** — All generated solutions, evaluation reports, and world model snapshots are persisted to disk for resumption and analysis.

## Architecture

```
k_search/
├── kernel_generators/
│   ├── kernel_generator.py             # Base LLM-driven kernel generator
│   ├── kernel_generator_world_model.py # World-model-aware generator (main loop)
│   ├── kernel_generator_prompts.py     # Prompt templates for generation/optimization
│   ├── world_model.py                  # World model data structures & JSON schema
│   ├── world_model_manager.py          # World model lifecycle (init/refine/select)
│   └── world_model_prompts.py          # World-model-injected prompt templates
├── tasks/
│   ├── task_base.py                    # Task protocol, Solution, EvalResult types
│   ├── prompts.py                      # Shared prompt blocks (CUDA XML format, etc.)
│   ├── cuda_kernel_task.py             # Multi-file CUDA kernel task
│   ├── cuda_kernel_eval.py             # CUDA kernel evaluator subprocess
│   ├── device_bench_task.py            # Unified device bench (CUDA/XPU Triton)
│   ├── device_bench/                   # Device bench evaluator
│   ├── cuda_bench/                     # CUDA bench evaluator
│   └── xpu_bench/                      # XPU bench evaluator
└── utils/
    ├── paths.py                        # Artifact directory management
    └── solution_db.py                  # Solution database (JSONL persistence)
```

## Quick Start

### Prerequisites

- NVIDIA GPU (H100/B200 recommended)
- An API key for an OpenAI-compatible LLM provider

### Installation

```bash
# Clone the repository
git clone https://github.com/caoshiyi/K-Search.git
cd K-Search

# Install dependencies
uv pip install openai wandb
```

### CUDA Kernel Optimization

Optimize a PyTorch reference model with multi-file CUDA kernels:

```bash
python generate_kernels_and_eval.py \
  --task-source cuda_kernel \
  --task-path path/to/reference.py \
  --model-name gpt-5.2 \
  --language cuda \
  --target-gpu H100 \
  --max-opt-rounds 20 \
  --world-model \
  --save-solutions
```

### Device Bench (Triton on NVIDIA / Intel XPU)

Optimize a PyTorch reference model with Triton kernels on any accelerator:

```bash
# NVIDIA CUDA
python generate_kernels_and_eval.py \
  --task-source device_bench \
  --task-path path/to/reference.py \
  --device-bench-device cuda:0 \
  --model-name gpt-5.2 \
  --language triton \
  --target-gpu H100 \
  --max-opt-rounds 20 \
  --world-model \
  --save-solutions

# Intel XPU
python generate_kernels_and_eval.py \
  --task-source device_bench \
  --task-path path/to/reference.py \
  --device-bench-device xpu:0 \
  --model-name gpt-5.2 \
  --language triton \
  --max-opt-rounds 20 \
  --world-model \
  --save-solutions
```

## CLI Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--task-source` | Task backend (`cuda_kernel`, `device_bench`, `xpu_bench`, `cuda_bench`) | `cuda_kernel` |
| `--task-path` | Path to reference .py file with Model class | *required* |
| `--model-name` | LLM model identifier | *required* |
| `--base-url` | OpenAI-compatible API base URL | OpenAI default |
| `--language` | Target language (`triton`, `cuda`) | `triton` |
| `--target-gpu` | Target GPU architecture hint | `H100` |
| `--max-opt-rounds` | Maximum optimization rounds | `5` |
| `--world-model` | Enable co-evolving world model | off |
| `--wm-stagnation-window` | Rounds without improvement before switching action | `5` |
| `--wm-max-difficulty` | Max action difficulty (1–5) to attempt | `4` |
| `--continue-from-solution` | Resume from an existing solution | — |
| `--continue-from-world-model` | Resume WM state (`auto` or path to JSON) | — |
| `--save-solutions` | Persist generated solutions to disk | off |
| `--artifacts-dir` | Base directory for all K-Search artifacts | `.ksearch` |
| `--wandb` | Enable Weights & Biases logging | off |
| `--cuda-kernel-num-correct-trials` | Number of correctness trials (cuda_kernel) | `5` |
| `--cuda-kernel-num-perf-trials` | Number of performance trials (cuda_kernel) | `100` |
| `--cuda-kernel-precision` | dtype for cuda_kernel eval | `fp16` |
| `--device-bench-device` | Device string (e.g. `cuda:0`, `xpu:0`) | auto |
| `--device-bench-precision` | dtype for device_bench eval | `fp16` |
| `--device-bench-num-correct-trials` | Number of correctness trials (device_bench) | `5` |
| `--device-bench-num-perf-trials` | Number of performance trials (device_bench) | `100` |

## Results

### FlashInfer-Bench

K-Search significantly outperforms state-of-the-art evolutionary search methods on complex kernels from [FlashInfer-Bench](https://bench.flashinfer.ai/), achieving an average **2.10×** improvement over OpenEvolve and up to **14.3×** on MoE kernels.

<p align="center">
  <img src="assets/main_figure.png" alt="K-Search FlashInfer-Bench Results" width="90%"/>
</p>

### GPUMode TriMul

K-Search achieves state-of-the-art performance on the [GPUMode TriMul](https://www.gpumode.com/home) task on H100 (**1028 µs**), surpassing both prior automated and human-designed solutions. The benchmark script (`results/gpumode_trimul/bench.sh`) evaluates kernels using the [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) upstream evaluator across 7 workload configurations, reporting per-benchmark latencies and geometric means with variance across multiple runs.

Geometric mean latency across 7 benchmarks (3 runs, H100, PyTorch 2.8.0+cu128, Triton 3.4.0):

| Submission ID | Leaderboard Score | Local Score | Std |
|---------------|-------------------|-------------|-----|
| **K-Search (ours)** | — | **1.028 ms** | 0.0007 ms |
| shiyegao | 1.074 ms | 1.067 ms | 0.0013 ms |
| TTT | 1.161 ms | 1.222 ms | 0.0018 ms |
| zeyushen | 1.140 ms | 1.240 ms | 0.0057 ms |

### Generated Kernels

We provide all generated kernels (K-Search, OpenEvolve, and ShinkaEvolve) under the `results/` folder for reproducibility and comparison:

| Task | Directory |
|------|-----------|
| MLA Paged Decode | `results/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1/` |
| MLA Paged Prefill | `results/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1/` |
| GQA Paged Decode | `results/gqa_paged/gqa_paged_decode_h32_kv4_d128_ps1/` |
| MoE FP8 | `results/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/` |
| GPUMode TriMul | `results/gpumode_trimul/` |

## Citation

If you find K-Search useful in your research, please cite our paper:

```bibtex
@article{cao2026k,
  title={K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model},
  author={Cao, Shiyi and Mao, Ziming and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2602.19128},
  year={2026}
}
```
