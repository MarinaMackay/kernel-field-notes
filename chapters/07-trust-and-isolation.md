# Trust boundaries, isolation, and worker state

## Knowledge first

An evaluator is an experiment runner and a trust boundary. Keep the reference, input generation, timers, result validation, and scoring outside candidate control. Reward hacking targets evaluator weaknesses; generalization failure concerns performance on new tasks. Neither can be ruled out by a single successful example.

Processes help contain crashes and isolate address spaces, but do not establish a complete security sandbox or guarantee stable GPU performance. Workers can retain contexts, modules, buffers, and allocator state. Useful persistence must be balanced against contamination and recovery. Distinguish queue, compile, execution, and infrastructure failures.

Multi-tenancy can contend for execution, memory, cache, and transfer resources. MPS and MIG provide different sharing/partitioning mechanisms with device-dependent support and limits. A partition's result must not be labeled as an equivalent whole-device result.

## Questions and answers

### 07.1 · Intermediate · Why validate outputs produced by the timed path?

**Before you answer — Correctness and timing can accidentally exercise different paths.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

A candidate could return valid results during a check yet use stale output, skip work, or launch unjoined side-stream work during timing. Tie source identity, inputs, output checks, and timing dependencies together, then independently re-evaluate finalists.

**Common mistake:** Assuming a warmup correctness pass proves all later executions are valid.

</details>

### 07.2 · Advanced · Should a device outage receive the same score as wrong arithmetic?

**Before you answer — A failure must be attributed before it becomes a reward.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. Record invalid infrastructure measurements separately, retry under a defined policy, and do not turn outages or nonpositive timing values into ordinary rewards. Candidate-caused failures still need clear compile/correctness/runtime categories.

**Common mistake:** Training the policy to interpret unreliable infrastructure as arithmetic feedback.

</details>

### 07.3 · Intermediate · Why might clearing a cache not release GPU memory?

**Before you answer — Persistent state includes objects, cached artifacts, and allocator pools.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Live references can retain allocations; an allocator can reserve memory beyond currently live tensors. Track allocated and reserved memory separately. A byte budget is more useful than an entry-count limit for unevenly sized artifacts, and a damaged execution context may require worker-process replacement.

**Common mistake:** Treating memory_reserved as the size of only live application tensors.

</details>

## Further reading

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
- [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/index.html)
- [NVIDIA MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html)
