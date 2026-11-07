# GPU foundations and the memory hierarchy

## Knowledge first

A GPU exposes many concurrent threads. A **streaming multiprocessor (SM)** contains execution resources, warp schedulers, registers, and shared-memory/cache resources. A CUDA core is an arithmetic execution resource, not an independent CPU-like processor assigned permanently to one software thread. Tensor Cores accelerate supported matrix operations; their supported formats depend on the architecture.

**Registers** hold thread-local working values. **Shared memory** is explicitly managed cooperation space. **L1/L2 caches** serve memory requests through hardware-managed policies. **Device DRAM** holds larger arrays; different products use HBM or GDDR. Local memory is a per-thread address space backed by device memory and can be cached—it does not mean on-chip storage.

A tensor has shape, dtype, device, and strides. A shape `(2, 3)` has six elements, not six bytes. FP32 storage for those elements is 24 bytes before allocator overhead. `randn` samples a normal distribution; `rand` samples a uniform distribution on `[0, 1)`.

## Questions and answers

### 01.1 · Foundation · Why not keep all model weights in registers?

**Before you answer — Capacity and bandwidth answer different questions.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Registers are a small working set shared among resident threads. Large parameter tensors need device memory; kernels bring portions nearer to execution as needed. More registers per thread can also reduce residency.

**Common mistake:** Treating the fastest memory as unlimited storage.

</details>

### 01.2 · Foundation · Are shared memory and L1 cache the same thing?

**Before you answer — An address space describes visibility; a cache describes how accesses are served.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. Software explicitly stores cooperating data in shared memory and must arrange synchronization. L1 caches accesses under hardware and instruction policies. Some architectures share physical storage capacity between these functions without making their semantics identical.

**Common mistake:** Assuming shared physical capacity makes the programming interfaces interchangeable.

</details>

### 01.3 · Foundation · Does a cache hit prove that memory access is coalesced?

**Before you answer — Hit rate and transaction efficiency measure different properties.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. A cache hit means data is found in a cache. Coalescing concerns the addresses requested across lanes by one memory instruction. Scattered accesses can hit cache; contiguous accesses can miss it.

**Common mistake:** Using cache-hit percentage as the only test of a good memory layout.

</details>

### 01.4 · Foundation · How do you address a transposed tensor?

**Before you answer — Strides are measured in elements for PyTorch tensors.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

For a two-dimensional tensor use `offset = i * stride_0 + j * stride_1`. A contiguous `(2, 3)` tensor normally has strides `(3, 1)`; its transpose has shape `(3, 2)` and strides `(1, 3)`. A kernel that hardcodes `i * N + j` promises contiguous layout only.

**Common mistake:** Assuming every reshape copies or every view is contiguous.

</details>

## Further reading

- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
