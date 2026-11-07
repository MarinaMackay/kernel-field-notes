# Kernel rewards and multi-turn learning pipelines

## Knowledge first

A kernel-learning loop can be written as: task → prompt → samples → immutable source snapshot → compile → correctness → benchmark → reward → trajectory → advantage → update → held-out evaluation. Configuration tuning selects parameters within a template; free-form optimization can change code, decomposition, and fusion.

Reward design should gate speed claims on valid measurements and correctness. An illustrative reward is zero for candidate compilation/correctness failure, and `1 + clip(log(speedup), -0.5, 1)` for a correct, valid measurement. Infrastructure failure is a separate invalid state. The logarithm expresses multiplicative improvement symmetrically before clipping, but the reward shape is a design choice, not a standard.

Multi-turn feedback can include compiler errors, numerical counterexamples, and profiling observations. Tools provide observations rather than sampled policy actions. Response masks control which tokens contribute to the action loss while retaining useful context. Train/validation/test separation must reflect task families, shapes, dtypes, and hardware, not just random seeds.

## Questions and answers

### 13.1 · Intermediate · What rewards does the illustrative formula give for S=1 and S=2?

**Before you answer — A speedup ratio must be positive and valid before taking its logarithm.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

`S=1` gives 1. `S=2` gives about 1.693 with the natural logarithm, before the upper clip is reached. Correct slow candidates have a floor of 0.5 in this design. Whether that incentive is desirable depends on the training problem; it does not repair noisy measurements.

**Common mistake:** Giving an unavailable device an ordinary zero reward or taking log of an invalid latency.

</details>

### 13.2 · Intermediate · How many evaluations are in 8 trajectories × 4 turns × 3 shapes?

**Before you answer — A trajectory, source version, and execution are different units.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

At most 32 source versions and 96 shape evaluations if every trajectory reaches every turn and each version is tested on all three shapes. Warmups, repeated timings, retries, compilation, and correctness cases can add more work. Preserve source hashes and policy versions.

**Common mistake:** Calling 96 evaluator executions 96 independent policy updates.

</details>

### 13.3 · Advanced · Should compiler-error tool messages be trained as policy actions?

**Before you answer — Masking a token’s loss is not removing it from context.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Usually not under an action-only policy loss: the model did not sample those tool tokens as its own actions. Keep the feedback visible in context, but apply response masks to the intended generated spans. Check tokenizer boundaries, truncation, and framework behavior explicitly.

**Common mistake:** Deleting the observations because their loss mask is zero.

</details>

### 13.4 · Advanced · Is changing only seeds an adequate held-out split?

**Before you answer — Validation influences decisions; an untouched test estimates final generalization.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Usually not for a claim about new operator families or hardware. Split along the dimensions of the intended generalization claim, freeze the final test, and report search budget. Track pass@1, pass@k under a defined sampling budget, correct/valid rates, independent runtime, and GPU-hours.

**Common mistake:** Using the same test repeatedly to choose rewards, prompts, and checkpoints while calling it held-out.

</details>

### 13.5 · Advanced · Why can faster evaluation fail to accelerate learning?

**Before you answer — Pipeline throughput is different from improvement per unit resource.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

More evaluations can be duplicates, invalid candidates, or uninformative equal-reward groups. Generation, update, transfers, or stragglers may dominate. Retaining only perfect trajectories may also remove examples of error recovery. Measure useful learning progress per wall time and resource budget.

**Common mistake:** Equating candidate count with policy improvement.

</details>

## Further reading

- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
- [KernelBench](https://arxiv.org/abs/2502.10517)
- [verl multi-turn training](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
