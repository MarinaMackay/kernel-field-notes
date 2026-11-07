# Measurement, streams, correctness, and profiling

## Knowledge first

A performance result is a claim about a specified workload, numerical contract, baseline, and measurement procedure. The knowledge sections below introduce each topic before its questions. Try answering before expanding the explanation. Public GPU results are attributed examples, not independent measurements for this book.

## A motivating example: is a kernel actually faster?

A benchmark can change which GPU kernel wins without changing either kernel.

In a public Triton report, the comparison below reverses after changing how the benchmark prepares the L2 cache:

| Cache preparation | cuBLAS BF16 | One Triton W8 configuration | Lower latency |
| --- | ---: | ---: | --- |
| Write-based flush | 46.1 μs | 43.0 μs | W8 |
| Read-based flush | 33.8 μs | 38.9 μs | cuBLAS |

The report attributes the difference to dirty cache lines: a subsequent kernel can pay for writing them back. The author tested A100 PCIe 40GB GPUs and provided reproducer code, profiling evidence, and limits. This is an author-reported result, not my independent reproduction, and it does not establish the same behavior on every architecture. See [Triton issue #12554](https://github.com/triton-lang/triton/issues/12554).

That example motivated this guide. A useful speedup claim needs a definition of the work, a valid timer, comparable conditions, and evidence that the result survives another run.

This article organizes my kernel-learning notes into explanations, worked examples, and questions to ask when reading or building a benchmark. It uses attributed public examples; it presents no new GPU measurements. Numerical examples labeled “illustrative” are calculations, not experimental results.

## 1. Start with the measurement contract

“Faster than PyTorch” is not enough information to reproduce a comparison.

Before comparing implementations, write down the operation, allowed inputs, numerical requirements, baseline, timed region, and execution mode. Keep eager PyTorch, warmed-up torch.compile, vendor libraries, and custom kernels identifiable.

Precision is part of the contract. A BF16 or TF32 computation is not automatically an equivalent replacement for a stricter FP32 task. Likewise, a contiguous-only implementation does not replace an operator promising arbitrary supported strides.

### The different clocks in a kernel project

| Quantity | What it answers |
| --- | --- |
| Device execution interval | How long does the selected GPU work take under this timing method? |
| Host launch overhead | How much host-side work is needed to submit operations? |
| Evaluation wall time | How long do preparation, compilation, checks, execution, and reporting take? |
| Search wall time | How long does candidate generation plus the evaluation process take? |
| Application latency | How long does the user-visible operation or request take? |
| Throughput | How many operations, candidates, requests, or tokens complete per unit time? |
| Goodput | How much valid, useful work completes per unit time under stated requirements? |

These quantities should not share an unlabeled “latency” column. An evaluation service can spend seconds compiling and checking a candidate whose kernel takes microseconds.

### 06.1 · Foundation · Can throughput improve while latency gets worse?

**Before you answer — Batching, throughput, latency, and goodput describe different objectives.**

<details markdown="1">
<summary>Reveal answer</summary>

Yes. A larger batch may finish more requests per second while each request waits longer. Define the unit, concurrency, and latency objective. In a kernel-search service, report valid and correct candidate rates alongside raw evaluation throughput.

</details>

### 06.2 · Foundation · Which baseline should I use?

**Before you answer — A baseline defines the meaning of a speedup.**

<details markdown="1">
<summary>Reveal answer</summary>

Use the implementation relevant to the claim. If the intended deployment uses torch.compile, an eager-only comparison leaves a question unanswered. Include both when useful, with their own warmup and configuration. Match allocation behavior, precision, supported inputs, and graph mode—or explicitly explain the difference.

</details>

### 06.3 · Foundation · Does changing precision count as a kernel optimization?

**Before you answer — Precision changes the numerical contract.**

<details markdown="1">
<summary>Reveal answer</summary>

It can be a useful system optimization, but report the changed numerical contract. Readers need to distinguish a faster implementation of equivalent requirements from a speed/accuracy trade-off.

</details>

## 2. Submission is not completion

A CUDA stream is an ordered queue of device work. The CPU can enqueue an operation and continue before the GPU finishes it. Operations on different streams can overlap when dependencies and hardware permit; concurrency is not guaranteed.

A host timestamp taken immediately after an asynchronous call may mostly describe submission.

The [PyTorch benchmark tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) provides a concrete example. Its unsynchronized timeit output reports about 25.3 μs for a warmed-up mul/sum operation, while torch.utils.benchmark reports about 232.9 μs for that operation. The tutorial explains synchronization and first-use effects. These are historical example outputs from its environment, not expected results for your hardware.

### Two timing boundaries

| Method | Boundary | Interpretation |
| --- | --- | --- |
| Unsynchronized host timer | CPU submits work, then reads the clock | Does not establish GPU completion |
| Synchronized host timer | Drain earlier work; start; submit; wait; stop | Host-observed interval including submission and waiting |
| CUDA events | Record events in a correctly ordered device stream | Elapsed device timeline interval; may include gaps and contention |

### 06.4 · Foundation · How are CUDA Event and perf_counter different?

**Before you answer — Host and device clocks observe different timelines.**

<details markdown="1">
<summary>Reveal answer</summary>

An event marks the device timeline. A host timer measures elapsed host time. Use synchronization before and after the host-timed region to exclude pending earlier work and wait for measured work. With events, wait for the ending event before reading elapsed_time; PyTorch reports milliseconds.

</details>

### 06.5 · Intermediate · Does a CUDA Event give pure kernel execution time?

**Before you answer — A device interval can include gaps and contention.**

<details markdown="1">
<summary>Reveal answer</summary>

Not unconditionally. A gap between events and work, other work in the measured stream, or contention can affect the interval. Tiny kernels can expose host submission gaps. Repeated launches or graph replay can help answer particular questions, but change the execution setup and must be labeled.

Speechmatics discusses these subtleties, with code and timeline illustrations, in [How to Accurately Time CUDA Kernels in PyTorch](https://blog.speechmatics.com/cuda-timings).

</details>

### 06.6 · Foundation · Why can .item(), printing, or copying to CPU change the timing?

**Before you answer — CPU-visible CUDA values can introduce synchronization.**

<details markdown="1">
<summary>Reveal answer</summary>

Reading a CUDA value on the CPU can force synchronization. Calls such as .item(), .tolist(), printing tensor values, and a blocking CPU copy may surface previously queued work. A suddenly slow line is not necessarily where the expensive computation originated.

</details>

### 06.7 · Intermediate · What if a candidate launches work on a side stream?

**Before you answer — A timing boundary must depend on every contributing stream.**

<details markdown="1">
<summary>Reveal answer</summary>

The ending event must depend on completion of every contributing stream. A timeline that ends before the side-stream output is ready can undercount the operation. Establish both input readiness before side-stream work and completion before the timing boundary.

</details>

### 06.8 · Intermediate · Are wait_stream(), record_stream(), and synchronize() interchangeable?

**Before you answer — Execution dependencies and allocation lifetimes are distinct.**

<details markdown="1">
<summary>Reveal answer</summary>

No. Stream waits establish dependencies for subsequently submitted work. Synchronization waits on the host. record_stream() informs the allocator about tensor use on another stream; it is not a substitute for an execution dependency. The stream rules and allocator interaction are documented in [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html).

</details>

### 06.9 · Intermediate · Does non_blocking=True guarantee a transfer overlaps computation?

**Before you answer — Asynchronous submission does not guarantee overlap.**

<details markdown="1">
<summary>Reveal answer</summary>

No. Useful overlap depends on the transfer, memory type, streams, dependencies, and device capabilities. Pinned host memory can enable efficient asynchronous transfers, but the flag alone does not prove overlap. Verify the intended timeline.

</details>

### 06.10 · Intermediate · Should I benchmark with CUDA_LAUNCH_BLOCKING=1?

**Before you answer — Debugging settings can change benchmark behavior.**

<details markdown="1">
<summary>Reveal answer</summary>

Use it for debugging asynchronous failures when appropriate, not as an unnoticed benchmark setting. It changes execution behavior. Also distinguish a GPU block barrier such as __syncthreads() from a host device synchronization: they operate at different scopes.

</details>

## 3. Warmup, compilation, and graphs answer different questions

The first call may initialize a CUDA context, load libraries or modules, compile a specialization, allocate memory, or run autotuning. A steady-state measurement deliberately answers a different question from startup latency.

Report compilation/autotuning, first invocation, and steady state separately when each matters. Warm up the actual shape, dtype, layout, and mode being measured; one warmup input does not establish that every later specialization is already compiled.

### 06.11 · Intermediate · Is a cold worker the same as a cold cache?

**Before you answer — First-use work, persistent workers, and cache state are separate.**

<details markdown="1">
<summary>Reveal answer</summary>

No. A worker can have an initialized context and compiled modules while the relevant data is absent from cache. Conversely, a warm worker can retain buffers, allocator state, and other state that affects evaluation. Describe initialization state and data-cache state separately.

</details>

### 06.12 · Intermediate · How do torch.compile and CUDA Graphs differ?

**Before you answer — Compilation and graph replay change different costs.**

<details markdown="1">
<summary>Reveal answer</summary>

Compilation can transform an operation graph and generate or select implementations, including fusion. CUDA Graph replay reduces repeated submission overhead for a captured workflow; it does not by itself turn the underlying kernels into faster algorithms. Some compile configurations use graph replay when applicable. Record what actually happened, including graph breaks or recompilation.

</details>

### 06.13 · Intermediate · What should I check if graph replay is still slow?

**Before you answer — Capture reuses execution assumptions.**

<details markdown="1">
<summary>Reveal answer</summary>

Check whether the time includes input copies, repeated capture, dynamic shapes, fallback regions, or work outside the graph. Static buffers also make address reuse explicit. Compare ordinary launches and replay separately rather than giving one implementation a different mode without saying so.

</details>

### 06.14 · Intermediate · What do benchmark helpers handle, and what do I still own?

**Before you answer — Specialization and tuning determine which code runs.**

<details markdown="1">
<summary>Reveal answer</summary>

PyTorch's benchmark Timer handles warmup and accelerator synchronization when needed. It also controls CPU thread settings, which can matter in mixed workloads. blocked_autorange selects a block size to amortize measurement overhead and collects repeated measurements.

Triton's do_bench returns milliseconds and exposes warmup, repetition, quantiles, and summary modes. Its current documentation describes warmup and rep as time budgets in milliseconds, not counts. Specify median or requested quantiles explicitly instead of assuming defaults. You still own the operation contract, cache policy, stream dependencies, environment, and output validation. See [PyTorch benchmark utilities](https://docs.pytorch.org/docs/2.15/benchmark_utils.html) and [Triton do_bench](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html).

Autotuning executes candidates. For kernels that mutate buffers, reset or restore state between trials; otherwise, tuning can change the input to later trials. Record tuning cost and cache keys. A compiled-cache hit is not proof that a new input was validated.

</details>

## 4. “Cold cache” needs a procedure

GPU data can be served through caches rather than device DRAM. Reusing the same buffers may produce a different workload from cycling through a larger working set.

Choose the experiment to match the application. Record working-set size, address reuse, attempted eviction, and what belongs inside the timing boundary. Keep policy comparable for the baseline and candidate.

### 06.15 · Intermediate · Should every benchmark flush L2?

**Before you answer — Repeated addresses may reuse cache contents.**

<details markdown="1">
<summary>Reveal answer</summary>

No. Warm reuse can be realistic; streaming fresh data can also be realistic. Neither is a universal default for every application. If you deliberately evict, describe the procedure and check the library version rather than assuming every helper has identical cache behavior.

</details>

### 06.16 · Intermediate · Can preparation outside the timer affect the interval inside it?

**Before you answer — Cache preparation is part of an experimental setup.**

<details markdown="1">
<summary>Reveal answer</summary>

Yes. Preparation can change cache contents, dirty state, clocks, or allocator state. The opening Triton report illustrates why “outside the timed region” does not mean “without consequences.” Treat its architecture-specific evidence as a reason to inspect the setup, not a universal correction factor.

</details>

### 06.17 · Intermediate · Are cache hits and coalesced accesses the same thing?

**Before you answer — Cache reuse, coalescing, and banks describe different mechanisms.**

<details markdown="1">
<summary>Reveal answer</summary>

No. Cache behavior concerns where requested data is found. Coalescing concerns how addresses requested by lanes are serviced in memory transactions. Identical useful bytes do not guarantee identical transferred bytes. Shared-memory bank conflicts are a separate issue again.

This distinction matters when interpreting effective bandwidth: state which bytes you counted and which level of the memory system you mean.

</details>

### 06.18 · Intermediate · Do fixed clocks remove all noise?

**Before you answer — Hardware operating conditions can change observed latency.**

<details markdown="1">
<summary>Reveal answer</summary>

No. Clock controls, when available and permitted, help control one variable. Thermal conditions, power limits, competing processes, and the benchmark sequence still matter. Record the clock policy and observed conditions; a configured limit is not evidence that every run sustained it.

</details>

## 5. Statistics: a winning sample is a hypothesis

For the illustrative samples 5, 5, 5, 5, and 30 μs, the minimum and median are 5 μs, while the mean is 10 μs. Each statistic answers a different question.

Store raw measurements and explain their sampling unit. A percentile across averages of many launches is not the same as a percentile across individual request latencies. Five samples are not a convincing basis for a tail-latency claim.

### 06.19 · Intermediate · Why interleave A/B measurements?

**Before you answer — Measurement order can correlate with drift.**

<details markdown="1">
<summary>Reveal answer</summary>

Always measuring A first and B later can correlate implementation with temperature, clocks, or system load. Alternating or randomized paired blocks can reduce order bias. An A–B–B–A sequence is one practical pattern; it does not fix unequal warmup or cache policy.

</details>

### 06.20 · Advanced · Why independently retest the winner?

**Before you answer — Searching many noisy candidates introduces selection bias.**

<details markdown="1">
<summary>Reveal answer</summary>

Search selects from noisy observations. Trying many configurations increases the opportunity to select one that benefited from favorable noise. Freeze the source and configuration, then repeat baseline and winner measurements in a fresh confirmation stage.

A candidate observed at 1.201× but subsequently measured at 1.196×, 1.202×, and 1.197× has not demonstrated a stable margin above 1.20×.

</details>

### 06.21 · Foundation · Why use the geometric mean of speedups?

**Before you answer — Multiplicative ratios need an appropriate aggregation.**

<details markdown="1">
<summary>Reveal answer</summary>

For positive task speedups S_i, the geometric mean is exp(mean(log(S_i))). An illustrative 2× improvement and 0.5× regression give a geometric mean of 1×; the arithmetic mean gives 1.25×.

The geometric mean does not describe total application runtime, and it does not tell you how to score failures. Report task coverage and the failure policy explicitly.

</details>

### 06.22 · Intermediate · How does speedup differ from fast_p?

**Before you answer — Task-level speedup and suite-level success rate differ.**

<details markdown="1">
<summary>Reveal answer</summary>

A task's speedup is baseline time divided by candidate time. KernelBench's fast_p summarizes the fraction of tasks whose candidates are correct and exceed the speed threshold p. Failed tasks remain part of the suite denominator. Reporting only successful-task speedups can hide how often a method fails. State the baseline and suite version. See [KernelBench](https://github.com/ScalingIntelligence/KernelBench).

</details>

### 06.23 · Advanced · Does trying more candidates mean the model learned?

**Before you answer — Search and learning are different procedures.**

<details markdown="1">
<summary>Reveal answer</summary>

Not necessarily. Best-of-N selection or iterative prompting can improve the selected program without changing model weights. In RL, noisy or invalid measurements also affect the learning signal. Keep candidate failures separate from infrastructure-invalid evaluations, and keep model selection separate from held-out testing. Changing only a random seed does not establish generalization across task families, shapes, dtypes, or hardware.

</details>

## 6. Correctness is part of the performance claim

Before timing, define what counts as the same operation. For a numerical comparison, a common form is:

$$
|y-y_{\mathrm{ref}}| \leq \mathrm{atol}+\mathrm{rtol}|y_{\mathrm{ref}}|.
$$

This formula does not choose the tolerance for you. Base it on the dtype, operation, accumulation, and application. Near zero, absolute tolerance matters; at larger magnitudes, the relative term grows.

Check supported shape, dtype, stride, layout, and boundary cases. Test tails that are not multiples of a tile, small and empty inputs when supported, sign and magnitude variation, repeated indices where relevant, and the specified NaN/Inf behavior.

### 06.24 · Intermediate · Why isn't one random input enough?

**Before you answer — A test suite samples a declared input contract.**

<details markdown="1">
<summary>Reveal answer</summary>

It can miss unwritten output elements, indexing assumptions, numerically difficult inputs, and race conditions. Initialize fully written output buffers with a recognizable poison value, compare full outputs, and preserve input copies when mutation is forbidden. A matching sum can conceal compensating errors.

</details>

### 06.25 · Foundation · Does allclose mean bitwise equality?

**Before you answer — Numerical closeness is not bitwise equality.**

<details markdown="1">
<summary>Reveal answer</summary>

No. Floating-point addition is not associative, so a changed reduction order can change low bits. Bitwise equality, numerical closeness, and deterministic execution are distinct requirements. An assert_close check can also enforce shape, dtype, and device expectations depending on its options.

</details>

### 06.26 · Intermediate · Is a race just ordinary nondeterminism?

**Before you answer — Race freedom and deterministic rounding are distinct.**

<details markdown="1">
<summary>Reveal answer</summary>

No. A valid floating-point atomic reduction can vary with addition order while meeting its numerical contract. An unsynchronized conflicting access can be a correctness bug. Repeated successful runs do not prove its absence. NVIDIA's [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) provides memory, race, initialization, and synchronization checks with documented scopes; no single check proves arbitrary candidate code correct.

</details>

### 06.27 · Intermediate · Why validate the benchmark execution path?

**Before you answer — Checking one path does not validate another path.**

<details markdown="1">
<summary>Reveal answer</summary>

Generated code can behave differently during a separate correctness check and the timed call. Validate outputs from the same execution path used for measurement, with validation work outside the timed interval and a stated sampling policy. Fresh inputs, output buffers, addresses, and processes can reveal unwanted reuse or state dependence.

</details>

### 06.28 · Advanced · How is reward hacking different from overfitting?

**Before you answer — The evaluator must enforce its own trust boundary.**

<details markdown="1">
<summary>Reveal answer</summary>

Overfitting means doing well on observed tasks and poorly on new ones. Reward hacking exploits the evaluator or metric: for example, returning stale results, changing timer behavior, or leaving work outside the measured dependency chain. The categories can overlap.

Keep reference computation, input generation, timing, and scoring under trusted evaluator control. Static scans alone are not sufficient. The [Kevin paper](https://arxiv.org/html/2507.11948v1) discusses reward-hacking problems in generated-kernel evaluation.

</details>

## 7. Profile to explain a valid result

Use a timeline to locate the expensive region, then inspect the relevant kernels. [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) helps relate CPU activity, GPU kernels, copies, and stream gaps. [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) examines kernel-level behavior such as memory traffic, registers, occupancy, and stalls.

Profiling can perturb execution. Metric collection can replay work or control cache and clocks. A profiled duration and an ordinary benchmark duration are different observations unless you establish comparable conditions.

### 06.29 · Intermediate · How do I tell memory-bound from compute-bound?

**Before you answer — Arithmetic intensity relates work to data movement.**

<details markdown="1">
<summary>Reveal answer</summary>

Start with an estimate of arithmetic intensity:

$$
AI=\frac{\mathrm{FLOPs}}{\mathrm{Bytes}}, \qquad
P \leq \min(P_{\mathrm{peak}}, B_{\mathrm{mem}}AI).
$$

Specify the precision and memory level used in that model. For illustrative FP32 vector addition, two reads and one write imply 12 useful bytes and one floating-point addition per element: AI ≈ 1/12 FLOP/byte. This suggests a bandwidth limit for large streaming inputs, not a complete model of launch overhead or cached tiny inputs.

Low compute and memory utilization do not uniquely diagnose “latency-bound.” Inspect dependencies, available parallelism, launch gaps, access patterns, and the meaning of the counters. NVIDIA discusses performance analysis and its limits in the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

</details>

### 06.30 · Intermediate · Why can a bigger tile or higher occupancy make a kernel slower?

**Before you answer — A tile trades reuse against resource demand.**

<details markdown="1">
<summary>Reveal answer</summary>

A larger tile can improve reuse while using more registers and shared memory, reducing residency or causing spills. Higher occupancy can help hide latency, but is not an objective by itself. Instruction-level parallelism, synchronization, and useful work still matter. Compare theoretical resource-limited occupancy with achieved occupancy, then test a specific hypothesis.

</details>

### 06.31 · Intermediate · What do stall reasons tell me?

**Before you answer — Profiling counters require architecture and metric context.**

<details markdown="1">
<summary>Reveal answer</summary>

They suggest where eligible progress is being delayed, not an automatic fix. Long Scoreboard often points toward dependencies on L1TEX-serviced operations; Short Scoreboard toward MIO-related dependencies such as shared memory; Barrier toward synchronization. Math-pipeline pressure and Not Selected need context. Names and interpretations are architecture/tool dependent.

Do not optimize the largest stall percentage without checking whether it limits overall progress. Return to an ordinary benchmark after the change.

</details>

### 06.32 · Intermediate · Does fusion always help?

**Before you answer — Fusion trades intermediate traffic for a changed kernel.**

<details markdown="1">
<summary>Reveal answer</summary>

It can reduce launches and intermediate traffic, but can increase register pressure, synchronization, or code complexity, and may replace a better specialized implementation. Test the full region. A locally faster kernel can leave the application unchanged or slower.

</details>

## 8. The machine and evaluator are part of the experiment

An initialized worker saves startup overhead but retains state. Two subprocesses can share physical GPU resources even when their address spaces and CUDA contexts differ.

### 06.33 · Advanced · Does subprocess isolation make timings independent?

**Before you answer — Process, performance, and security isolation differ.**

<details markdown="1">
<summary>Reveal answer</summary>

No. Processes can contend for SMs, cache, memory bandwidth, and copy engines. A subprocess helps contain some failures and supports worker recovery; it is not automatically a secure sandbox for arbitrary generated code.

For controlled kernel latency, avoid unrelated GPU work or use a deliberately specified partition. Document MPS or MIG configuration. [MPS](https://docs.nvidia.com/deploy/mps/latest/index.html) supports CUDA multiprocess sharing and resource controls; [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html) partitions supported hardware resources. A partition's score should not be silently compared with a whole-device score.

</details>

### 06.34 · Advanced · What should happen after a timeout, OOM, or device failure?

**Before you answer — Failure attribution is part of an evaluation result.**

<details markdown="1">
<summary>Reveal answer</summary>

Classify queue timeout, compilation timeout, execution timeout, initialization failure, and device failure separately. An unavailable GPU is not evidence that the candidate is numerically wrong. After a context-poisoning error, retire the affected worker rather than trusting subsequent results.

Retain enough information to retry infrastructure-invalid evaluations without silently rewarding or penalizing a candidate for a machine fault.

</details>

### 06.35 · Intermediate · Why can memory_reserved stay high after tensors are freed?

**Before you answer — Cache capacity must account for bytes and lifetimes.**

<details markdown="1">
<summary>Reveal answer</summary>

Allocator reservations and live tensor allocations describe different things. References and allocator pools can retain memory. Record relevant memory state and apply a deliberate worker-reuse policy instead of assuming every trial starts identically. Limit caches by bytes as well as entry count when objects vary greatly in size.

</details>

### 06.36 · Intermediate · If a job took five seconds, was the kernel running for five seconds?

**Before you answer — Queues couple throughput to waiting time.**

<details markdown="1">
<summary>Reveal answer</summary>

Not necessarily. Separate queue wait, compilation, initialization, checks, device execution, and reporting. Admission control, backpressure, concurrency limits, and stragglers affect service latency and throughput.

Under stable long-run conditions, Little's Law relates average in-system jobs L, arrival rate λ, and average time in system W: L = λW. An illustrative rate of 2 jobs/s and 10 s average sojourn time gives 20 jobs in the system, including waiting and service. It does not measure kernel execution.

</details>

## 9. Translate a kernel result into a useful decision

### 06.37 · Foundation · Why doesn't a 2× faster kernel make the whole model 2× faster?

**Before you answer — Amdahl’s law limits end-to-end benefits.**

<details markdown="1">
<summary>Reveal answer</summary>

In an illustrative 100 ms application, a 20 ms component reduced to 10 ms yields 90 ms overall: about 1.11×. Under Amdahl's simplified fixed-work assumptions,

$$
S_{\mathrm{total}}=\frac{1}{(1-f)+f/s}.
$$

With f = 0.2, even eliminating that component entirely caps the overall speedup at 1.25×. Real deployments can also change batching, memory pressure, communication, and scheduling, so measure end-to-end behavior as well.

</details>

### 06.38 · Intermediate · What if production latency suddenly becomes noisy?

**Before you answer — Online latency includes more than kernel execution.**

<details markdown="1">
<summary>Reveal answer</summary>

Check request mix, batch sizes, long contexts, KV-cache pressure, preemption, competing GPU work, clocks, host scheduling, and garbage collection before concluding that kernel code regressed. For serving, distinguish time to first token from inter-token latency and define which percentile and load you report.

</details>

### 06.39 · Intermediate · How would I investigate a kernel that is slower than expected?

**Before you answer — Diagnosis begins by verifying the observation.**

<details markdown="1">
<summary>Reveal answer</summary>

1. Recheck output validity, the baseline, precision, timed region, synchronization, warmup, and cache policy.
2. Estimate useful work, traffic, and a relevant performance bound.
3. Inspect the application timeline for launch gaps, copies, and serialization.
4. Inspect the kernel for resource use, access patterns, spills, dependencies, and contention.
5. Change one hypothesis-driven factor.
6. Recheck correctness and repeat the ordinary comparison, then independently confirm the selected result.

</details>

### 06.40 · Advanced · How would I design a trustworthy evaluation service?

**Before you answer — An evaluation service needs a measurement contract and failure model.**

<details markdown="1">
<summary>Reveal answer</summary>

Use an admission-controlled queue, GPU workers, and a narrow candidate interface. Keep evaluator logic outside candidate control. Separate compilation work from controlled timing windows; use an explicit worker initialization and reuse policy.

Generate reference inputs, preserve originals, validate supported cases, measure with correct dependencies, and inspect outputs from the benchmark path. Classify failures and recover damaged workers. Store source hash, configuration, environment, inputs or generation recipe, checks, raw timings, and confirmation results.

For agent feedback, return useful compiler errors or numerical counterexamples rather than only “failed.” Track valid rate, correctness, unique candidates, and GPU-hours alongside throughput. If more GPUs become available, first identify whether generation, compilation, evaluation, or training is the actual bottleneck.

</details>

## A reporting checklist to copy

- [ ] Operation, supported inputs, layout, and numerical contract.
- [ ] Baseline implementation, precision, allocation behavior, and compilation/graph mode.
- [ ] Hardware, partition/sharing configuration, software versions, and source hash.
- [ ] Exact timing boundary and every contributing stream dependency.
- [ ] Startup, compilation/autotuning, first-call, and steady-state costs kept distinguishable.
- [ ] Warmup procedure, repetitions or time budget, units, and sampling unit.
- [ ] Buffer reuse, cache preparation, clock policy, and competing workloads.
- [ ] Edge-case and mutation checks, tolerances, full-output comparison, and failures.
- [ ] Raw samples, median, variability, aggregation, task coverage, and failure policy.
- [ ] Independent confirmation using frozen code and fresh evaluation state.
- [ ] Application-level impact, limitations, and instructions or links for reproduction.

A speedup is useful when another reader can identify the work, reconstruct the conditions, and understand where the claim stops applying.

## Appendix: an explicit same-stream timing sketch

This educational example keeps allocation and validation outside both intervals. It reuses the same buffers, does not deliberately evict caches, uses ordinary launches, and runs the two methods in separate phases. It is not a controlled comparison between different kernels. The phases and per-sample synchronization can change conditions; do not interpret their difference as an exact launch-overhead measurement.

The script has been checked for Python syntax but has not been executed on a GPU for this article. It reports no expected timing values. For tiny operations, the event interval can still include submission gaps; use a purpose-built helper or an explicitly specified graph experiment when that is the question.

```python
import statistics
from time import perf_counter
import torch

if not torch.cuda.is_available():
    raise SystemExit("This example requires a CUDA-capable PyTorch environment.")

a = torch.randn(1_000_003, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
out = torch.empty_like(a)

def operation():
    torch.add(a, b, out=out)

for _ in range(100):
    operation()
torch.cuda.synchronize()
torch.testing.assert_close(out, a + b)

# One operation per sample, same-stream, repeated-buffer scenario.
# Input/output allocation, validation, and warmup are outside the intervals.
host_ms, event_ms = [], []
for _ in range(30):
    torch.cuda.synchronize()
    t0 = perf_counter()
    operation()
    torch.cuda.synchronize()
    host_ms.append((perf_counter() - t0) * 1e3)

# Pre-create and initialize event objects outside the measured intervals.
pairs = [(torch.cuda.Event(enable_timing=True),
          torch.cuda.Event(enable_timing=True)) for _ in range(30)]
for start, end in pairs:
    start.record()
    end.record()
torch.cuda.synchronize()

for start, end in pairs:
    start.record()
    operation()
    end.record()
    end.synchronize()
    event_ms.append(start.elapsed_time(end))

torch.testing.assert_close(out, a + b)
print("host-observed median ms:", statistics.median(host_ms))
print("event-interval median ms:", statistics.median(event_ms))
```

## Sources and evidence notes

The two numerical public examples are attributed above. All other numerical worked examples are illustrative. No new GPU experiment was run for this article, and the timing sketch or reasoning here should not be treated as a hardware-specific performance guarantee.

- [Triton #12554: cache preparation and ranking reversal](https://github.com/triton-lang/triton/issues/12554). Author report opened October 30, 2026; open when checked for this article. Its findings are limited to the reported environment.
- [PyTorch benchmark tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).
- [Speechmatics: How to Accurately Time CUDA Kernels in PyTorch](https://blog.speechmatics.com/cuda-timings), March 28, 2023.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html), [benchmark utilities](https://docs.pytorch.org/docs/2.15/benchmark_utils.html), and [torch.compile](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html).
- [Triton do_bench API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html).
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) and [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) and [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/html/2507.11948v1).
- [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/latest/index.html) and [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html).

Written from personal learning notes with AI assistance for organization, English editing, and source lookup.

