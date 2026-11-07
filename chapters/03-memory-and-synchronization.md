# Memory access and synchronization

## Knowledge first

Look at addresses across lanes of the **same instruction** when reasoning about coalescing. Alignment, access width, and address spacing determine transaction demand. Structure-of-arrays layouts often help when neighboring threads need the same field from neighboring records. Vectorized loads need appropriate alignment and supported types.

Shared-memory accesses have bank behavior; bank conflicts are different from global-memory coalescing. Padding or swizzling can change bank mapping. Warp shuffles exchange register values among participating lanes. Participation masks and active lanes are part of correctness.

Synchronization has scope: a block barrier coordinates a block, a warp primitive coordinates participating lanes, a memory fence orders relevant memory operations, and host synchronization waits for device work. They are not substitutes for one another. Independent thread scheduling makes implicit warp lockstep an unsafe general synchronization assumption.

## Questions and answers

### 03.1 · Intermediate · Can each thread read consecutive elements yet produce poor coalescing?

**Before you answer — Same-thread sequential access is not the definition of coalescing.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Yes. If each lane starts a distant chunk, the first load instruction touches widely separated addresses even though each lane advances sequentially. Examine the cross-lane pattern for each instruction.

**Common mistake:** Checking only the loop inside one thread.

</details>

### 03.2 · Intermediate · Why is a transpose tile often declared as float tile[32][33]?

**Before you answer — Bank mapping depends on access pattern, word size, and architecture.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

In a common 32-bank, four-byte-word model, padding changes the row stride so a column read no longer sends distinct words to the same bank. It is a technique for that access pattern, not a universal prescription for all types or layouts. Same-address broadcast is a separate case.

**Common mistake:** Adding one column everywhere without analyzing the actual mapping.

</details>

### 03.3 · Foundation · Can __syncthreads() be inside if (threadIdx.x < 16)?

**Before you answer — A barrier must be reached consistently by the participating group.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Not as a block-wide barrier when other live block threads skip it. Arrange a uniform barrier and predicate the data work instead. A warp-scoped primitive can be appropriate only when its participation requirements are satisfied.

**Common mistake:** Treating a barrier as a local wait for only the calling thread.

</details>

### 03.4 · Intermediate · Does atomicAdd synchronize every block?

**Before you answer — An atomic operation is indivisible for its target operation.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. It does not make every block arrive at a common point or establish a general grid barrier. A separate kernel launch in an appropriately ordered stream is often a simple way to separate dependent global phases.

**Common mistake:** Confusing an atomic update with global phase completion.

</details>

### 03.5 · Advanced · How should I approach cp.async, TMA, warp specialization, and newer Tensor Core instructions?

**Before you answer — Asynchronous copy and matrix features have architecture-specific contracts.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

First identify the target architecture and supported instruction set. Then specify producer/consumer roles, buffers, arrival/completion barriers, and storage lifetimes. Ampere asynchronous copies, Hopper TMA/warpgroup features, and Blackwell tensor-memory instructions require different programming contracts; an architecture-specific binary is not automatically portable.

**Common mistake:** Copying a specialized kernel to another target merely because both GPUs support CUDA.

</details>

## Further reading

- [CUDA advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
