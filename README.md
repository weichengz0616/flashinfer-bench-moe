# [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)

## Update: the final evaluation results

Average speedup vs. flashinfer_wrapper_9sdjf3 (flashinfer v0.6.9): **1.62x** (#5; #1 speedup is 1.71x).

| [Workload](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/blob/main/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl) uuid | Input length | Latency(ms) |
| ------ | ------ | ------ |
| e05c6c03 | 1 | 0.068 |
| 2e69caee | 15 | 0.088 |
| b8f4f012 | 7 | 0.105 |
| 8cba5890 | 14 | 0.157 |
| a7c2bcfd | 16 | 0.159 |
| f7d6ac7c | 52 | 0.182 |
| 5eadab1e | 62 | 0.188 |
| eedc63b2 | 59 | 0.218 |
| 6230e838 | 32 | 0.225 |
| 76010cb4 | 54 | 0.233 |
| 81955b1e | 55 | 0.234 |
| fc378037 | 53 | 0.236 |
| 74d7ff04 | 57 | 0.237 |
| e626d3e6 | 58 | 0.242 |
| 4822167c | 56 | 0.254 |
| 8f1ff9f1 | 80 | 0.281 |
| 1a4c6ba1 | 901 | 0.327 |
| 58a34f27 | 11948 | 0.855 |
| 5e8dc11c | 14107 | 1.200 |

[MoE kernel entrypoint](https://github.com/weichengz0616/flashinfer-bench-moe/blob/main/solution/triton/main.py#L2669)

---

Create high-performance GPU kernels for state-of-the-art LLM architectures on NVIDIA Blackwell GPUs with humans and/or AI agents.

---

<p align="center">
  <a href="https://www.nvidia.com"><img src="images/nvidia-logo.svg" alt="NVIDIA" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://modal.com"><img src="images/modal-logo.png" alt="Modal" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mlsys.org"><img src="images/mlsys-logo.svg" alt="MLSys" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer"><img src="images/flashinfer-logo.png" alt="FlashInfer" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer-bench"><img src="images/fib_logo.png" alt="FlashInfer-Bench" height="50"/></a>
</p>

---

[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) is our official framework to evaluate your AI-generated kernels.

## Updates

* 2026.02.05: Full dataset for definitions and workloads are released at [HuggingFace](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)

## Competition Tracks

The competition features three tracks, each targeting a critical LLM operation:

| Track | Description |
|-------|-------------|
| **fused_moe** | Fused Mixture-of-Experts kernel for efficient expert routing and computation |
| **sparse_attention** | Sparse attention mechanisms for long-context inference |
| **gated_delta_net** | Gated delta network operations for efficient state updates |

**Fork this template once per track** you want to compete in (separate repos for each track).

## Getting Started

### 1. Fork This Template

Click "Use this template" or fork this repository to create your solution repo.

### 2. Install Dependencies

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

### 3. Download the TraceSet

We provide kernel definitions and workloads in [FlashInfer-Trace format](https://bench.flashinfer.ai/docs/flashinfer-trace). Clone the competition dataset from HuggingFace:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

Set the environment variable:

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### 4. Configure Your Solution

Edit `config.toml` to set your track and team info:

```toml
[solution]
name = "my-team-solution-v1"      # Solution name
definition = "fused_moe"          # Track: fused_moe | sparse_attention | gated_delta_net
author = "team-name"              # Team/author name

[build]
language = "triton"               # triton | cuda
entry_point = "kernel"            # Kernel function name
```

### 5. Implement Your Kernel

**For Triton:**
Edit `solution/triton/kernel.py` with your implementation.

**For CUDA:**
Edit `solution/cuda/kernel.cu` and `solution/cuda/binding.py` with your implementation.

## Development Workflow

### Pack Your Solution

Generate `solution.json` from your source files:

```bash
python scripts/pack_solution.py
```

### Run Local Benchmarks

Test your solution on your local GPU:

```bash
python scripts/run_local.py
```

Requires: Local CUDA-capable GPU and `FIB_DATASET_PATH` environment variable.

### Run Cloud Benchmarks (Modal)

Test your solution on NVIDIA B200 GPUs via Modal:

**One-time setup:**

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace
```

**Run benchmark:**

```bash
modal run scripts/run_modal.py
```

## Submission

To submit your solution for evaluation:

1. Ensure your implementation is complete and tested
2. Run `python scripts/pack_solution.py` to generate `solution.json`
3. Commit and push your changes
4. Tag your commit for evaluation (e.g., `git tag submission-v1`)

## Project Structure

```
flashinfer-bench-starter-kit/
├── README.md                    # This file
├── config.toml                  # Track configuration (edit this)
├── solution/                    # Solution source files
│   ├── triton/                  # Triton implementation
│   │   └── kernel.py           # Your Triton kernel
│   └── cuda/                    # CUDA implementation
│       ├── kernel.cu           # Your CUDA kernel
│       └── binding.py          # TVM FFI bindings
├── scripts/                     # Utility scripts
│   ├── run_local.py            # Local benchmark runner
│   ├── run_modal.py            # Modal cloud benchmark runner
│   └── pack_solution.py        # Pack source files into solution.json
└── images/                      # Sponsor logos
```

## Additional Resources

### FlashInfer Trace Viewer

FlashInfer Trace consists of multiple JSON objects (definitions, workloads, solutions, and traces), which can contain large code blocks. To easily visualize and inspect these objects, you can use the [FlashInfer Trace Viewer](https://bench.flashinfer.ai/viewer). Simply paste any FlashInfer Trace JSON into the viewer to get a friendly, structured view of its contents.

### Solution Handling API

```python
from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files, extract_solution_to_files

# Pack source files into a Solution object
spec = BuildSpec(
    language="triton",  # or "cuda"
    target_hardware=["cuda"],
    entry_point="my_kernel",
)
solution = pack_solution_from_files(
    path="./my_solution_dir",
    spec=spec,
    name="my_solution_v1",
    definition="fused_moe",
    author="your_name",
)

# Extract a Solution to files in a working directory
extract_solution_to_files(solution, "./output_dir")
```

### Running Sanitizers

```python
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

output = flashinfer_bench_run_sanitizer(
    solution=solution,
    workload=workload,
    sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
    timeout=300,
)
print(output)
```

### NCU Profiling

```python
from flashinfer_bench.agents import flashinfer_bench_run_ncu

output = flashinfer_bench_run_ncu(
    solution=solution,
    workload=workload,
    set="detailed",
    page="details",
    timeout=120,
)
print(output)
```

### List Available Tools

```python
from flashinfer_bench.agents import get_all_tool_schemas

schemas = get_all_tool_schemas()
# Returns list of OpenAI-compatible function schemas
```

## Notes

### Destination Passing Style (DPS)

FlashInfer-Bench uses destination passing style (DPS) by default, where both inputs and outputs are passed as function parameters. DPS avoids measuring tensor allocation overhead, resulting in more accurate performance numbers. We recommend using DPS when possible, as it yields better benchmark results.

**Important:** Avoid using variadic input arguments in your kernel signatures, as they will fail the builder validation check.

If your kernel uses value-returning style (i.e., returns output tensors instead of writing to pre-allocated ones), set `destination_passing_style` to `false` in your solution's `spec`:

```json
{
  "name": "my_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "triton",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.py::my_kernel",
    "dependencies": [],
    "destination_passing_style": false
  },
  "sources": [...]
}
```

**Common error when DPS is mismatched:**

```
Destination-passing style callable: expected xx parameters, but got xx
```

This can happen for two reasons: (1) your kernel function signature has the wrong number of parameters, or (2) your kernel uses value-returning style but the solution still has `destination_passing_style` set to `true` by default. For the latter case, fix by setting `destination_passing_style` to `false`.

### CUDA Kernel Bindings

For CUDA kernel implementations, we recommend using [TVM FFI](https://tvm.apache.org/ffi/) for Python bindings. The `flashinfer_bench.agents` module provides TVM FFI agent instruction prompts to assist with development.

You can set the `binding` field in your solution's `spec` to specify the C++ binding type. Defaults to `"tvm-ffi"` if not specified. Supported values: `"tvm-ffi"`, `"torch"`.

```json
{
  "name": "my_cuda_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "cuda",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.cu::my_kernel",
    "dependencies": [],
    "binding": "torch"
  },
  "sources": [...]
}
```
