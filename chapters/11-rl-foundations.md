# Reinforcement learning: concepts before formulas

## Knowledge first

A policy selects actions given observations or state. A trajectory records actions and outcomes; return aggregates rewards, often with discount `gamma`. A value function predicts expected return; advantage compares an action against a baseline. Code-generation feedback can be partially observed because cache, load, or hidden evaluator state is not fully visible.

SFT imitates selected examples. Best-of-N searches samples without changing policy weights. RL updates policy parameters from sampled outcomes. Rejection-sampling fine-tuning selects outcomes and then trains on them; name the actual learning procedure rather than calling every agent loop RL.

For an autoregressive response, sequence log probability is the sum of conditional token log probabilities. Policy gradients differentiate these probabilities, not the CUDA runtime. A state-dependent baseline that does not depend on the sampled action can reduce variance without changing the expected score-function gradient under the usual assumptions.

## Questions and answers

### 11.1 · Foundation · Is a multi-turn coding agent automatically doing RL?

**Before you answer — Repeated attempts do not necessarily update a model.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

No. It may be inference-time search: generate, run tools, read feedback, and try again. It becomes a learning procedure when selected data or rewards drive parameter updates. Report sampling budget and weight updates separately.

**Common mistake:** Calling prompt editing or retrying a policy-gradient update.

</details>

### 11.2 · Intermediate · Can a high reward have a low advantage?

**Before you answer — Reward and advantage use different reference points.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

Yes. Advantage measures improvement over the relevant baseline. A reward of 10 can be disappointing where the baseline is 12, while reward 1 can be strong where the baseline is 0. The estimator and normalization determine the numerical scale.

**Common mistake:** Assuming positive reward must increase every action probability.

</details>

### 11.3 · Intermediate · How can timing feedback train a code policy?

**Before you answer — The sampled action likelihood is differentiable even when execution is not.**

<details markdown="1">
<summary>Reveal answer and common mistake</summary>

A score-function estimator weights gradients of token log probabilities by returns or advantage estimates. The evaluator supplies a scalar outcome; it need not supply a derivative of runtime with respect to source code. Reliable rewards, sampling-policy likelihoods, and masks are essential.

**Common mistake:** Attempting to backpropagate through a compiler or physical stopwatch as if it were a neural layer.

</details>

## Further reading

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath: original GRPO formulation](https://arxiv.org/abs/2402.03300)
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
