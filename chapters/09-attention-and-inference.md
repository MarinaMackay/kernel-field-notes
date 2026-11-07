# Attention, KV caches, and inference

## Knowledge first

Prefill processes prompt tokens; autoregressive decode adds new tokens over successive steps. Their matrix shapes and data reuse differ, so their bottlenecks depend on batch, sequence length, implementation, and hardware. Small-batch decode often exposes weight/KV bandwidth; this is a workload tendency, not a law for every configuration.

A KV cache stores attention keys and values. For a conventional cache, bytes are approximately `2 × batch × tokens × layers × KV_heads × head_dim × bytes_per_element`. GQA shares KV heads among groups of query heads; MQA uses one KV head. Prefix caching reuses compatible previously computed prefix state. Paging changes allocation and addressing; quantization changes storage and numerical behavior.

FlashAttention tiles attention and maintains online normalization to reduce intermediate HBM traffic. PagedAttention addresses KV-cache memory management and access. They solve different problems and compatible implementations can combine their ideas. Continuous batching lets completed requests leave and new work enter between scheduling steps; chunked prefill partitions long prompt work.

## Questions and answers

### 09.1 · Foundation · FlashAttention or PagedAttention—which one replaces the other?

**Before you answer — Compare the resource being optimized.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Neither is a general replacement. FlashAttention targets the IO behavior of attention computation; PagedAttention uses block-based KV storage and mapping to reduce allocation waste and enable flexible reuse. An inference stack can use both.

**Common mistake:** Calling both simply the same attention speedup.

</details>

### 09.2 · Intermediate · Estimate KV storage for 32 layers, 8 KV heads, head_dim=128, and BF16.

**Before you answer — The KV formula needs KV heads, not query heads.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Per token per sequence: `2 × 32 × 8 × 128 × 2 = 131072 bytes = 128 KiB`. At 8192 tokens this is 1 GiB; 32 such sequences require 32 GiB of tensor payload. Allocation slack, metadata, quantization scales, and sharding can change actual per-device storage.

**Common mistake:** Using the number of query heads for a GQA model or confusing GB with GiB.

</details>

### 09.3 · Advanced · Derive the online softmax update.

**Before you answer — A running normalizer must rescale when its reference maximum changes.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Let old maximum be `m`, old sum be `l = sum_old exp(x-m)`, and new block maximum be `m_b`. Set `m_new = max(m,m_b)`. Then `l_new = exp(m-m_new)*l + sum_block exp(x-m_new)`. The first factor expresses the old sum in the new coordinate system. Attention also rescales its accumulated weighted-value numerator. Producing every standalone softmax probability still requires retaining inputs or another pass; a running normalizer alone does not output all probabilities immediately.

**Common mistake:** Claiming that maintaining max and sum alone emits every normalized output in one streaming pass.

</details>

### 09.4 · Intermediate · What does 16 GB of weights divided by 3.35 TB/s tell us?

**Before you answer — A bandwidth bound is a model, not a measured prediction.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

It gives about 4.78 ms for that traffic under an illustrative ideal-bandwidth assumption. It excludes KV traffic, other operations, launch/communication costs, and inefficiency; cache reuse or changed traffic can alter the model. It is not an observed token latency or a guaranteed speed after quantization.

**Common mistake:** Promising a 4× end-to-end speedup from 4× smaller weight storage.

</details>

### 09.5 · Intermediate · Why separate prefill and decode, or use speculative decoding?

**Before you answer — Scheduling changes affect both waiting time and interference.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Disaggregation can reduce interference but adds KV transfer and coordination costs. Speculation uses a draft proposal and a target verification procedure to accept multiple tokens; preservation of the target distribution requires the appropriate exact acceptance/correction algorithm. Neither method is automatically beneficial for every workload.

**Common mistake:** Assuming any draft-and-accept shortcut preserves the original model distribution.

</details>


### 09.6 · Foundation · Are GQA, MQA, and prefix caching the same idea?

**Before you answer — Head sharing changes model/cache structure; prefix reuse changes which existing computation can be reused.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

MQA uses a single KV head shared across query heads. GQA uses multiple KV heads, each shared by a group of query heads. Prefix caching reuses compatible cached state for a previously computed matching prefix; it is not a head-sharing architecture. Cache validity depends on the model and execution context, not only matching visible text. MLA uses a different compressed latent attention representation and needs its own storage accounting.

**Common mistake:** Applying the ordinary uncompressed KV formula unchanged to every attention architecture.

</details>

## Further reading

- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
