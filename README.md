# research-neural-net
Repository for neural network research 

## Chapters

- **Ch.1** — One fact: single node, single point. Model memorizes the one example flawlessly, knows nothing else.
- **Ch.2** — A dataset: single node, many points. Forced to compromise, first real generalization.
Chapters still ahead (the actual historical/pedagogical path):
- **Ch.3** — Multiple features: real data isn't one number, it's several (x1, x2, ...) — weight becomes a vector.
- **Ch.4** — The wall: a single neuron, no matter how many features, can only draw a straight boundary. Some patterns are mathematically impossible for it (XOR) — this is the actual historical dead-end that killed neural nets for a decade.
- **Ch.5** — Hidden layers: the fix for Ch.4 — stack neurons to bend that straight line into a curve (you've already built this).
- **Ch.6** — Scale: Python loops break down at real sizes → matrices/vectorization.
- **Ch.7** — Vanishing gradients (you already found this) → ReLU.
- **Ch.8** — Better optimizers, regularization, mini-batches → the actual LLM training recipe.
