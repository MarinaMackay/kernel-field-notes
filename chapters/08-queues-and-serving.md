# Queues, scheduling, and service objectives

## Knowledge first

Admission control decides which work enters a service. Backpressure slows producers when downstream capacity is exhausted. FIFO can create head-of-line blocking; strict priorities can starve low-priority work. Fairness requires a unit: number of tasks, device time, or another stated resource.

For a stable system, Little's law relates averages: `L = arrival rate × mean time in system`. Use consistent boundaries and units. Queue time is not execution time. More concurrency can hide gaps, but can also increase contention, memory pressure, and tail latency.

Serving metrics need a workload: request sizes, arrival pattern, concurrency, batching, and latency objectives. For generation, distinguish time to first token (TTFT), inter-token latency (ITL), and completed useful tokens or requests per second.

## Questions and answers

### 08.1 · Foundation · At 2 requests/s and 10 s mean time in system, how many requests are present?

**Before you answer — Little’s law concerns averages over the same system boundary.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

`L = 2 × 10 = 20` requests on average, assuming a stable system and matching definitions. This is not a maximum queue length and does not by itself give P95 latency.

**Common mistake:** Substituting a percentile into a mean-value identity.

</details>

### 08.2 · Intermediate · What happens if generation is faster than kernel evaluation?

**Before you answer — Every stage has its own capacity and queue.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Pending work grows unless the producer is slowed or work is rejected. Bound the queue, expose queue age, and use admission/backpressure thresholds. Separate compilation capacity from controlled timing capacity when they interfere.

**Common mistake:** Increasing rollout concurrency indefinitely because the generator has spare compute.

</details>

### 08.3 · Advanced · What should I measure before adding twice as many GPUs?

**Before you answer — A straggler changes the completion of a group.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Measure generation, compilation, correctness, timing, update, transfer, and queue delays; valid/correct rates; GPU utilization; and group completion tails. Add capacity to the bottleneck, then remeasure. If one candidate stalls an entire group, timeout or scheduling policy may matter more than raw device count.

**Common mistake:** Assuming doubling one stage doubles end-to-end learning or serving throughput.

</details>

## Further reading

- [Little’s law — original paper DOI](https://doi.org/10.1287/opre.9.3.383)
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
