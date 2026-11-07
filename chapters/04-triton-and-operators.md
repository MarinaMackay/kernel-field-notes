# Triton, reductions, normalization, and GEMM

## Knowledge first

Triton expresses operations on blocks of data. `tl.program_id` identifies a program; `tl.arange` creates logical offsets. A logical block size is not a CUDA thread count. The compiler maps data and operations to the selected target with launch settings and layouts.

For a one-dimensional add, form `i = program_id * BLOCK + arange(0, BLOCK)`, mask with `i < n`, load the two inputs, add, and store. Dtype, strides, masking, aliasing, and numerical requirements are still the author's responsibility.

Reductions combine many inputs. Floating-point addition is not associative, so changing reduction order can change rounding. Stable softmax subtracts the row maximum before exponentiation. RMSNorm divides by `sqrt(mean(x²) + epsilon)` and applies weights; it does not subtract the mean as LayerNorm does. FP32 accumulation often improves low-precision reduction accuracy.

GEMM computes `C[i,j] = sum_k A[i,k] B[k,j]`, about `2MNK` FLOPs. A tiled implementation selects an output tile, loops over K tiles, accumulates, and stores. Reuse competes with resource use. Fusion can save launches and intermediate traffic; autotuning trades search cost for steady-state performance.

## Questions and answers

### 04.1 · Foundation · Does BLOCK_SIZE=256 mean 256 CUDA threads?

**Before you answer — A Triton program operates on a logical data block.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. It means 256 logical elements in this indexing example. Physical execution also depends on the target, compiler mapping, layout, and launch configuration such as `num_warps`. Do not infer thread count from element count.

**Common mistake:** Equating `tl.arange` length with the number of threads.

</details>

### 04.2 · Foundation · For n=10 and BLOCK=4, which offsets are valid?

**Before you answer — Tail masks protect memory accesses, not just the final output.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Three programs form offsets 0–3, 4–7, and 8–11. The final two offsets must be masked on loads and stores. Choose masked fallback values appropriate to subsequent operations.

**Common mistake:** Masking the store while still reading beyond an input.

</details>

### 04.3 · Intermediate · Why use negative infinity for masked softmax inputs?

**Before you answer — A reduction needs neutral values for inactive entries.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

For a valid row with finite elements, negative infinity cannot increase the maximum and contributes zero after exponentiation. Zero is not a neutral value for max when every valid logit is negative. An entirely masked row needs an explicit semantic choice; subtracting negative infinity from itself produces NaN.

**Common mistake:** Assuming the same mask fallback is correct for sum, max, and softmax.

</details>

### 04.4 · Intermediate · Is a tree reduction always bitwise deterministic?

**Before you answer — Determinism depends on a specified order and execution contract.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No universal guarantee follows from the word tree. A fixed reduction structure on a fixed implementation can provide repeatability, while compiler transformations, precision modes, and architecture changes can alter results. Atomic floating-point accumulation can vary in order without being a data race when used correctly.

**Common mistake:** Promising reproducibility across devices because the algorithm is called a tree.

</details>

### 04.5 · Intermediate · What does a minimal GEMM plan need?

**Before you answer — Tiling reuses loaded operands across multiple outputs.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Select tile coordinates from program IDs; create row, column, and K offsets; mask edges; load A and B tiles using their strides; accumulate compatible dot products, commonly into FP32; advance along K; convert/store under the output contract. Split-K adds partial-result reduction; persistent schedules and grouped ordering change work assignment or cache reuse.

**Common mistake:** Assuming every `tl.dot` shape and dtype automatically selects the desired Tensor Core path.

</details>

### 04.6 · Intermediate · What can go wrong during autotuning?

**Before you answer — The configuration search actually runs candidate implementations.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Incorrect keys can reuse a configuration across workloads that need different choices. In-place kernels can mutate inputs during trials unless reset or restored. Too many stages can consume storage; more warps can increase overhead. Record compile/search cost and validate the chosen configuration.

**Common mistake:** Reporting the best steady-state microseconds while hiding minutes of one-time tuning.

</details>

### 04.7 · Intermediate · Should matmul, bias, and activation always be fused?

**Before you answer — Fusion removes boundaries but also changes resource use.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

An epilogue may avoid intermediate writes and additional launches, but a fused implementation can have higher register pressure or lose a superior library implementation. Compare equivalent numerical contracts and the entire operation. Combining independent small operations is a different fusion pattern.

**Common mistake:** Assuming fewer kernel launches proves lower total latency.

</details>


### 04.8 · Intermediate · How does RMSNorm differ from LayerNorm?

**Before you answer — RMS normalization uses a second moment without subtracting the mean.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

RMSNorm computes `y_i = x_i * w_i / sqrt(mean(x²) + epsilon)`. LayerNorm normalizes centered values using the mean and variance, with its specified affine transformation. For an RMSNorm kernel, accumulate the sum of squares with suitable precision, handle tails, and apply the weights under the dtype and stride contract. Low-precision input storage does not require low-precision accumulation.

**Common mistake:** Implementing mean subtraction and calling the result RMSNorm.

</details>

## Further reading

- [Triton vector addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
