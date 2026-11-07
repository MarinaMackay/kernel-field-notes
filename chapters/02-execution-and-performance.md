# Threads, warps, occupancy, and performance models

## Knowledge first

The hierarchy is **grid → blocks → warps → threads**. On NVIDIA GPUs, a warp has 32 lanes. A block is a cooperation scope and is scheduled onto one SM; multiple blocks may reside on an SM when resources allow. A block is not an SM. Do not assume an execution order between ordinary independent blocks.

**Arithmetic intensity (AI)** is FLOPs divided by bytes transferred at a stated memory boundary. A simple roofline model is `performance ≤ min(peak FLOP/s, bandwidth × AI)`. FP32 vector addition uses one add and ideally two reads plus one write: `1 / 12 FLOP/byte`. Tiny workloads may instead expose launch cost, limited parallelism, or dependencies.

**Occupancy** is resident active warps divided by the architectural maximum, not the fraction of arithmetic units doing useful work. Registers, shared memory, block limits, and launch size constrain it. More reuse and instruction-level parallelism may outperform a higher-occupancy configuration. A finite grid also creates partially filled final waves.

## Questions and answers

### 02.1 · Foundation · How many 256-thread blocks cover 1,000 elements?

**Before you answer — Global indexing combines a block index with an index inside that block.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

`ceil(1000 / 256) = 4` blocks launch 1,024 threads. The last 24 threads must avoid out-of-range accesses, for example with `if (i < n)`. Each block contains eight warps.

**Common mistake:** Forgetting the tail because the allocated launch size exceeds the logical length.

</details>

### 02.2 · Intermediate · Work out occupancy for a hypothetical SM.

**Before you answer — Resource limits combine; the tightest limit wins.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Assume 65,536 registers, 64 maximum warps, 256 threads/block, and 64 registers/thread. One block uses 16,384 registers, so registers permit four blocks or 32 warps: 50% occupancy. If each block also needs 48 KiB and the available shared-memory budget is 96 KiB, only two fit: 16 warps or 25%. Real allocation granularity and other limits may reduce this further.

**Common mistake:** Presenting these hypothetical limits as specifications for every GPU.

</details>

### 02.3 · Intermediate · Why can a larger tile be slower?

**Before you answer — Tiling increases reuse but consumes finite working storage.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

A larger tile may increase register use, shared memory, spills, synchronization, or unused work at boundaries. It may reduce the number of resident blocks. Measure the relevant shapes rather than optimizing occupancy or tile size alone.

**Common mistake:** Assuming more reuse always outweighs every resource cost.

</details>

### 02.4 · Intermediate · Do low memory and compute throughput prove a kernel is latency-bound?

**Before you answer — Roofline is a bound at a chosen memory level, not a full latency simulator.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

They motivate further investigation. Check launch gaps, insufficient grid size, dependencies, synchronization, imbalance, and counter definitions. A low percentage alone does not identify the cause.

**Common mistake:** Assigning a bottleneck from two utilization percentages without a timeline.

</details>

### 02.5 · Intermediate · What is the tail effect with 133 blocks and 132 SMs?

**Before you answer — Grid size and device size interact.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Under the illustrative assumption that only one block fits per SM and all blocks take equal time, the first wave can occupy 132 SMs and the remaining block leaves most idle. Real scheduling and unequal block durations can change the shape of this tail.

**Common mistake:** Treating SM count alone as sufficient to predict runtime.

</details>

## Further reading

- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
