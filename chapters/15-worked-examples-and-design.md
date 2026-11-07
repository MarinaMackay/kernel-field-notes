# Worked estimates, code review, and system design

## Knowledge first

Back-of-the-envelope models are useful when assumptions are explicit. Use consistent units, count traffic at a named level, and separate a theoretical lower bound from a measured result. For a dense GEMM with no input rereads and one output write, ideal bytes are approximately `(MK + KN + MN) × element_bytes`; accumulation precision and extra reads can change the real traffic.

For code review, begin with the input contract and participating threads. A grid-stride loop decouples grid size from input size. Warp reductions require valid lane participation. A block-level partial reduction can then use shared memory and a final stage. Transpose kernels need both efficient global access and appropriate shared-memory mapping.

Open-ended answers should state assumptions, decompose the system, identify bottlenecks/failure modes, and propose observations that distinguish hypotheses. Good design answers include what would change the decision, rather than a list of tools alone.

## Questions and answers

### 15.1 · Intermediate · Estimate FLOPs and ideal traffic for a 4096³ BF16 GEMM.

**Before you answer — A theoretical count is not a hardware utilization measurement.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

`2*4096³ = 137438953472` FLOPs. With two input matrices and one output, all BF16 and counted once, payload is `3*4096²*2 = 100663296` bytes. The resulting ideal AI is about 1365 FLOP/byte. Compare with the matching precision’s ridge point, but account for actual reuse, extra traffic, and implementation before predicting runtime.

**Common mistake:** Using dense BF16 compute peak for a different numerical mode or treating the ideal traffic count as observed HBM traffic.

</details>

### 15.2 · Advanced · What assumptions does a full-mask warp reduction need?

**Before you answer — Shuffle masks describe participation, not permission to read nonexistent values.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Every named participating lane must execute the matching collective consistently, and every source value used must be valid. A full-mask reduction sketch usually assumes complete warps and compatible control flow. A final partial warp needs correct masks and a reduction structure that does not depend on invalid lanes. Also initialize any atomic output and account for block-level accumulation order.

**Common mistake:** Making a reduction generic merely by adding `if (i < n)` around a load.

</details>

### 15.3 · Advanced · A kernel is slower than expected. What is your investigation order?

**Before you answer — The observation must be verified before explaining its cause.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Verify contract, baseline, warmup, stream dependencies, and timing validity. Estimate work/traffic and inspect a system timeline for launch gaps or transfers. Profile the kernel for resource limits, memory behavior, dependencies, and stalls. Change one supported hypothesis and rerun clean timings. Choose cache conditions appropriate to the workload rather than requiring a flush in every case.

**Common mistake:** Optimizing the first suspicious counter without reproducing the slowdown.

</details>

### 15.4 · Advanced · Design a kernel evaluation service.

**Before you answer — A service has a control plane, workers, data, and failure boundaries.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Define the task schema and numerical contract; retain immutable source and input metadata; use bounded queues and admission control; compile and execute under explicit limits; test diverse inputs and timed-path outputs; isolate/recover workers; control timing contention; separate candidate from infrastructure failures; preserve raw samples and environment versions; independently recheck finalists. Document the sandbox threat model separately from performance isolation.

**Common mistake:** Treating a subprocess as the complete security and performance design.

</details>

### 15.5 · Advanced · Online token latency suddenly spikes. What do you inspect?

**Before you answer — A latency regression can originate outside the kernel.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Correlate queue wait, prompt/context lengths, batch composition, KV usage/eviction, preemption, transfers, neighbors, clock/thermal state, CPU scheduling, and allocator/runtime activity with the spike. Separate TTFT from per-token latency and compare matched workloads. Use the evidence to select a targeted change.

**Common mistake:** Attributing a P95 serving regression to arithmetic throughput before examining workload and scheduling changes.

</details>

## Further reading

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
