# About this book

Kernel Field Notes is a free, question-led learning book about GPU programming and AI systems. It organizes personal study notes into explanations, worked calculations, questions, answers, and primary references. It makes no claim of a new research result or independent GPU benchmark.

## How to read

Read **Knowledge first** before attempting the questions. Each question includes a specific concept reminder. Try answering, then expand the answer. Foundation questions test vocabulary and simple models; Intermediate questions test mechanisms and trade-offs; Advanced questions test assumptions, derivations, debugging, and design. These levels are editorial guidance, not standardized interview ratings.

The structure is inspired by question-led technical books, including [Chip Huyen's ML Interviews Book](https://github.com/chiphuyen/ml-interviews-book). The prose and question answers here were assembled from the author's own notes with AI editing assistance; this is not a copy or an official extension of that book.

## Coverage of the first edition

This is version 0.1: 15 chapters and 102 answered questions. It is a substantive first edition, not a claim that every original note has completed translation and verification.

| Original note area | Current treatment | Remaining depth |
| --- | --- | --- |
| GPU structure and memory | Chapter 1 | Original embedded images not reused; no universal hardware-size table |
| Execution, roofline, occupancy | Chapters 2–3 | Architecture-specific limits and advanced instruction examples need separate versioned treatment |
| Streams, events, asynchronous copies | Chapter 6, questions 4–10 | More executable multi-stream examples |
| Benchmarking, statistics, profiling | Chapter 6, all 40 original article questions preserved | GPU execution and independent reproduction remain unperformed |
| Triton, strides, reductions, softmax, RMSNorm, GEMM, fusion, tuning | Chapters 1 and 4 | Full runnable operator implementations and unit tests are not supplied yet |
| Compiler/JIT, integration, graphs | Chapters 5–6 | IR walkthroughs and version-pinned integration examples |
| Correctness, numerical error, races | Chapter 6, questions 24–27; chapters 4 and 15 | Executed sanitizer examples |
| Reward hacking and trust | Chapters 6–7 | Adversarial harness tests and threat-model implementation |
| Workers, residency, timeouts, MPS/MIG | Chapters 6–7 | Device-specific partition and recovery experiments |
| Admission, queues, backpressure, fairness | Chapters 6 and 8 | Trace-driven scheduling comparisons |
| Serving, attention, KV, batching | Chapter 9 | Detailed MLA, FP8 KV, Flash-Decoding, and engine-specific examples |
| RL, PPO/GRPO, variants | Chapters 11–12 | Full loss derivations, implementation-specific masks, and variant comparisons |
| Kernel rewards and multi-turn training | Chapter 13 | Executed end-to-end training configuration |
| Papers, frameworks, public projects | Chapter 14 | Individual recent-paper reading records and current API/CLI checks |
| Estimates, code questions, distributed training, open design | Chapters 10 and 15, with related questions in other chapters | Complete executable CUDA reduction/transpose and Triton GEMM exercises |

The private editorial inventory retains the full original source notes, links, image references, and unresolved details. It is intentionally separate from this public repository: original Notion attachment URLs and unreviewed images are not distributed. No specific private-project architecture is described here.

The remaining-paper queue from the notes includes CUDA Agent, SOL-ExecBench, daVinci-kernel, KernelOPT, Harness Engineering, and additional kernel-learning projects. Their note-level claims are not presented as verified paper results in this edition. Exact statistics, endpoints, model names, and revision-specific behavior must be checked against their original sources before a dedicated reading note is added.

## Evidence and reproducibility

- Worked estimates are calculations under explicit assumptions, not observed measurements.
- GPU timings quoted in Chapter 6 are attributed author-reported public examples. They have not been independently reproduced for this book.
- The timing appendix is educational. Python syntax and the static book were checked; CUDA execution was not.
- Hardware features and APIs depend on versions and targets. Linked documentation can change; use the versions matching your environment.
- There are no tracking scripts, subscriptions, or account requirements in the generated book.

## Corrections

Please open a [GitHub issue](https://github.com/MarinaMackay/kernel-field-notes/issues) with the chapter/question ID, the disputed statement, a counterexample or primary reference, and the relevant hardware/software version. Suggestions for missing knowledge or questions are welcome too.
