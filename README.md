# Kernel Field Notes

**GPU Programming & AI Systems — Concepts, Questions, and Worked Examples**

A free, question-led web book: learn a concept, try a question, reveal the explanation, and follow primary references. Version 0.1 contains **15 chapters and 102 answered questions**, with Foundation, Intermediate, and Advanced labels.

[Read the web book](https://marinamackay.github.io/kernel-field-notes/) · [Browse the Markdown chapters](chapters/) · [Coverage and limitations](ABOUT.md) · [Suggest a correction](../../issues)

The web URL becomes available after GitHub Pages is enabled and deployment succeeds. Until then, the Markdown chapters are directly readable here for free.

## Reading path

1. [GPU foundations and memory](chapters/01-gpu-foundations.md)
2. [Execution and performance models](chapters/02-execution-and-performance.md)
3. [Memory access and synchronization](chapters/03-memory-and-synchronization.md)
4. [Triton and operators](chapters/04-triton-and-operators.md)
5. [Compilation and graphs](chapters/05-compilers-and-graphs.md)
6. [Measurement, streams, correctness, and profiling — 40 Q&A](chapters/06-measurement-and-validation.md)
7. [Trust, isolation, and workers](chapters/07-trust-and-isolation.md)
8. [Queues, scheduling, and service objectives](chapters/08-queues-and-serving.md)
9. [Attention and inference](chapters/09-attention-and-inference.md)
10. [Distributed execution and training](chapters/10-distributed-and-training.md)
11. [RL foundations](chapters/11-rl-foundations.md)
12. [PPO, GRPO, and variants](chapters/12-ppo-grpo-and-variants.md)
13. [Kernel learning pipelines](chapters/13-kernel-learning-pipelines.md)
14. [Reading papers and tools](chapters/14-reading-papers-and-tools.md)
15. [Worked estimates and system design](chapters/15-worked-examples-and-design.md)

This is a living first edition. It does not claim to include every detailed implementation or paper review from the original study notes. See [ABOUT.md](ABOUT.md) for an explicit topic-by-topic accounting. No independent GPU benchmark or novel research result is claimed.

## Build and preview

Requires Python 3.9+ and the pinned Markdown renderer:

```sh
python3 -m pip install -r requirements.txt
python3 tools/build.py
python3 tools/check.py
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. The generated `docs/` folder is self-contained and uses no CDN. Edit `chapters/*.md` and `ABOUT.md`, rebuild, and commit the source and generated pages together.

## GitHub Pages

In repository **Settings → Pages**, select **Deploy from a branch**, branch **main**, folder **/docs**, then Save. GitHub serves the generated static files without a paid plan for this public repository. The CI workflow checks that the committed site matches the Markdown source and that local links resolve.

## Attribution and contribution

Question-led organization was inspired by [Chip Huyen's ML Interviews Book](https://github.com/chiphuyen/ml-interviews-book). This book uses original educational prose assembled from personal notes, with AI editing assistance. Source references appear in each chapter; quoted/reported results remain attributed to their original authors.

See [CONTRIBUTING.md](CONTRIBUTING.md) for corrections and additions. No license for unrestricted reuse is asserted in this initial edition; linked third-party sources retain their own rights.
