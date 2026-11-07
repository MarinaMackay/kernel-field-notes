# PPO, GRPO, baselines, and policy lag

## Knowledge first

PPO uses a sampled-action likelihood ratio `rho = pi_new(a|s) / pi_old(a|s)`. Its clipped surrogate is `min(rho*A, clip(rho,1-epsilon,1+epsilon)*A)`. Clipping changes the incentive in the objective; it does not impose a hard global KL guarantee.

GRPO estimates relative advantages using a group of responses to the same problem instead of a separately trained value model in the original formulation. A common form is `(reward - group_mean) / (group_std + epsilon)`. Implementations differ in normalization, token/sequence aggregation, clipping, and reference regularization; record the actual variant.

A leave-one-out baseline excludes the current sample. A reference-model KL penalty and an old-policy likelihood ratio have different roles. Asynchronous rollouts introduce policy lag; training and inference can also disagree on token likelihoods. Version the policy and preserve the relevant sampling metadata.

## Questions and answers

### 12.1 · Intermediate · Compute group advantages for rewards [1,0,0,1].

**Before you answer — Specify population versus sample standard deviation.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

The mean is 0.5. Using population standard deviation gives 0.5, hence advantages `[1,-1,-1,1]` before epsilon. A sample-standard-deviation convention gives different magnitudes. An all-equal group has zero centered rewards and provides no relative reward signal under this estimator.

**Common mistake:** Leaving the standard-deviation convention unspecified in a worked example.

</details>

### 12.2 · Advanced · How does r_i - mean(other rewards) relate to r_i - group_mean?

**Before you answer — For G>1, leave-one-out centering changes scale.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Writing total reward as `G*mean`, the other-sample mean is `(G*mean-r_i)/(G-1)`. Subtraction gives `G/(G-1) * (r_i-mean)`. This relation concerns centering before any additional normalization.

**Common mistake:** Applying the relation to group size one or claiming it equates every full training objective.

</details>

### 12.3 · Intermediate · Do PPO clipping and a reference KL penalty do the same thing?

**Before you answer — Old-policy proximity and reference-policy proximity are distinct.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. The likelihood ratio compares to the policy used to sample that action. A reference KL term typically compares to a chosen reference model. Neither guarantees all desirable behavior; tune and monitor the actual implemented objective, distributions, and update statistics.

**Common mistake:** Calling the frozen reference model and the rollout policy the same object.

</details>

### 12.4 · Advanced · What should I check when reading DAPO or Dr. GRPO?

**Before you answer — Loss aggregation can alter length and difficulty weighting.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Compare the exact objectives and implementation: sampling/filtering of groups, upper/lower ratio clipping, reward standardization, token versus sequence aggregation, and length treatment. DAPO and Dr. GRPO motivate changes to these mechanisms; a variant name alone is insufficient to reproduce an update.

**Common mistake:** Explaining every long-response bias using only one normalization choice without inspecting the objective.

</details>


### 12.5 · Advanced · What if every response in a GRPO group has the same reward?

**Before you answer — Centered group rewards can lose the signal that distinguishes sampled actions.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Under the centered group estimator, all relative advantages are zero. A KL or other auxiliary term may still contribute to the update, so it does not follow that the full loss has no gradient. Inspect whether tasks are too easy, too hard, or collapsed to similar responses; check reward resolution and validity. Sampling, task selection, or exploration changes may help, but must be evaluated against held-out progress and cost.

**Common mistake:** Treating higher sampling temperature as a guaranteed fix for every zero-advantage group.

</details>

## Further reading

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath: original GRPO formulation](https://arxiv.org/abs/2402.03300)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO](https://arxiv.org/abs/2503.20783)
