# Compilation, JIT, integration, and CUDA Graphs

## Knowledge first

Source code is transformed into intermediate representations and target code. For NVIDIA targets, PTX is a virtual instruction set; SASS is architecture-specific machine code. A typical Triton compilation path passes through Triton/LLVM representations, but details vary by backend and compiler version. JIT compiles on demand; AOT prepares artifacts earlier.

Specialization can depend on compile-time constants, types, target, and compiler assumptions. A single Python function can therefore produce multiple artifacts. Cache keys must include the properties that affect generated code. A cache hit says an artifact can be reused, not that every new input has passed numerical validation.

`torch.compile` can capture and transform computation. CUDA Graphs record/replay a compatible execution graph to reduce repeated submission overhead. Framework integration also needs shape/dtype/device behavior, strides, aliasing, mutation, autograd, and compiler compatibility.

## Questions and answers

### 05.1 · Foundation · Are torch.compile and CUDA Graphs the same optimization?

**Before you answer — Code transformation and execution replay solve different problems.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. Compilation can change and fuse computation; graph replay primarily reduces repeated submission overhead. A framework mode may combine them. Report both the compilation setup and graph setup instead of treating them as one switch.

**Common mistake:** Claiming graph replay necessarily makes an individual arithmetic instruction faster.

</details>

### 05.2 · Intermediate · Why do graph examples reuse static input buffers?

**Before you answer — Capture retains execution assumptions and storage relationships.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Replay reuses captured work and its storage relationships. Copy new inputs into compatible captured buffers, or use supported framework mechanisms for changing inputs. Dynamic shapes or allocations can require separate captures or other handling. Always verify the current API contract.

**Common mistake:** Replacing a Python tensor variable and assuming a captured graph automatically follows it.

</details>

### 05.3 · Intermediate · Does opcheck prove that my custom operator computes the right answer?

**Before you answer — Operator registration and numerical validation are separate checks.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. Registration/metadata checks and integration checks do not replace comparison against a mathematical reference. Test numerical behavior, supported layouts, mutation/aliasing, and backward where promised; use gradient checks when appropriate.

**Common mistake:** Passing framework integration checks as the sole correctness evidence.

</details>

## Further reading

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
