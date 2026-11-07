# Distributed execution and training memory

## Knowledge first

Data parallelism replicates model computation over different data and communicates gradients. Sharded training partitions selected states. Tensor parallelism partitions layer computation; pipeline parallelism partitions stages; expert parallelism distributes experts and routes token work. Real systems combine these approaches, with collectives determined by the concrete algorithm.

Training memory includes weights, gradients, optimizer states, activations, temporary buffers, and communication storage. Mixed precision does not imply one universal bytes-per-parameter number. Activation checkpointing trades recomputation for fewer saved activations. BF16 has a wide exponent range but limited significand precision; FP16 training often needs loss scaling.

Communication time depends on message sizes, latency, topology, bandwidth, overlap, and contention. NVLink and PCIe specifications must be compared using the same directionality and aggregation convention. Tensor parallelism across nodes can be useful with suitable topology—it is not categorically forbidden.

## Questions and answers

### 10.1 · Foundation · Why is 7B BF16 inference not simply a 14 GB GPU requirement?

**Before you answer — Count each retained tensor and its precision.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

The weights alone are about 14 billion decimal bytes. Add KV cache, activations, workspaces, allocator overhead, and runtime/library state. Quantized weights add their own scales and implementation details. The required capacity depends on the workload.

**Common mistake:** Treating weight payload as the complete device-memory budget.

</details>

### 10.2 · Intermediate · Where does an illustrative 16 bytes per parameter come from?

**Before you answer — An optimizer memory estimate is an explicit accounting scenario.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

For one possible setup: BF16 weights 2 + BF16 gradients 2 + FP32 master weights 4 + FP32 Adam first moment 4 + FP32 second moment 4 = 16 bytes. Seven billion parameters give 112 GB before activations and other storage. Some implementations keep FP32 gradients, omit separate master copies, shard states, or use different optimizers, so recalculate rather than memorizing 16.

**Common mistake:** Calling 16 bytes an invariant of all mixed-precision Adam implementations.

</details>

### 10.3 · Intermediate · What is the ring all-reduce traffic per rank?

**Before you answer — Collective traffic formulas require an algorithm and a unit.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

For `p` ranks and an input buffer of `D` bytes, the usual ring reduce-scatter plus all-gather sends `2*(p-1)/p * D` bytes per rank, and receives the same amount. This excludes protocol overhead and does not mean runtime is independent of rank count: latency, topology, contention, and algorithm selection matter.

**Common mistake:** Adding send and receive totals and comparing them to a one-direction bandwidth specification.

</details>

### 10.4 · Intermediate · What should a DP / FSDP / TP / PP / EP comparison include?

**Before you answer — Partitioning trades memory or compute distribution for communication.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

State what is replicated or sharded, each collective and its frequency, activation transfers, memory peaks, and imbalance. DP commonly all-reduces gradients; FSDP-style strategies gather parameters and reduce-scatter gradients; TP communicates within layers; PP transfers stage activations and can have bubbles; EP routes tokens and can suffer expert imbalance. Exact communication patterns depend on the implementation.

**Common mistake:** Reducing every parallelism strategy to a single universal collective.

</details>

## Further reading

- [PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
