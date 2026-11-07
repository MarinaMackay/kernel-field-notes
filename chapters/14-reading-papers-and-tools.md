# Reading papers and choosing public tools

## Knowledge first

A useful reading note separates the task, workload, baseline, correctness contract, evaluation conditions, and claimed contribution. A fast result against one software baseline does not establish proximity to hardware limits. A faster isolated subkernel does not establish model-level correctness or end-to-end benefit.

For agent work, distinguish the **agent harness** (files, tool calls, context, retries) from the **evaluation harness** (reference, input generation, correctness, timing, isolation, reporting). A training framework supplies orchestration and updates; it cannot infer whether a CUDA result is numerically correct without an evaluator.

Start with a concrete workload and the source revision. KernelBench provides tasks for evaluating generated kernels; public competition and application-trace projects provide different workloads. CUTLASS/CuTe, CUDA, Triton, and other DSLs provide different levels of mapping control. Read the actual implementation and license before reusing code. Version-specific CLI commands and paper statistics belong in separately verified reading records.

## Questions and answers

### 14.1 · Intermediate · What does fast_p tell me that an average speedup does not?

**Before you answer — A suite metric needs both a success rule and a denominator.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

It is the fraction of suite tasks that satisfy correctness and exceed a stated speedup threshold under the benchmark definition. Failed tasks remain relevant to coverage. A mean over only successful tasks can hide low success rates. Verify strictness of the threshold and baseline in the benchmark version.

**Common mistake:** Comparing fast_p across different task sets or baselines as if they were identical experiments.

</details>

### 14.2 · Advanced · How should I study a kernel evaluation service such as KernelGYM?

**Before you answer — An environment’s architecture and current API are separate claims.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Trace client request → API/queue → GPU worker → compile/check/time → result. Identify timeout ownership, process lifetime, source identity, and error categories. Then inspect a pinned revision for exact endpoints and configuration; do not rely on unversioned notes as an API reference.

**Common mistake:** Copying endpoint names and training paths from an old note without checking the current repository.

</details>

### 14.3 · Intermediate · What should a paper comparison table record?

**Before you answer — A reference implementation supplies semantics, not necessarily the best baseline.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Record source/revision, date, task set, hardware and precision, baseline mode, correctness, timing/cache setup, search budget, aggregate denominator, application validation, and whether the result was independently reproduced. Mark unavailable or unverified details explicitly.

**Common mistake:** Retelling an abstract as if every number had been independently checked.

</details>

## Further reading

- [KernelBench](https://arxiv.org/abs/2502.10517)
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
- [KernelGYM](https://github.com/hkust-nlp/KernelGYM)
- [GPU MODE kernelbot](https://github.com/gpu-mode/kernelbot)
- [FlashInfer Bench](https://github.com/flashinfer-ai/flashinfer-bench)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
