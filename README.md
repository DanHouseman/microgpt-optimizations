# microgpt-optimizations
Optimizations to Kaparthy's microgpt.py with test harness for benchmarking.

# microgpt vs microgpt_optimized

This repository contains two pure-Python, from-scratch character-level GPT implementations:

- `microgpt.py` — the original “microGPT” (micrograd-style scalar autograd + tiny GPT).
- `microgpt_optimized.py` — an optimized variant focused on **computation efficiency** and **memory efficiency** while staying **pure Python**.
- `bench_compare.py` — a benchmark harness to compare the two across the specific areas that changed.

The models train on Andrej Karpathy’s `makemore` `names.txt` dataset (downloaded automatically to `input.txt` on first run).

## Quick start

### Run original training/inference
```bash
python microgpt.py
```
Run optimized training/inference

```bash
python microgpt_optimized.py
```

Benchmarking: bench_compare.py

What it measures

bench_compare.py compares microgpt.py vs microgpt_optimized.py in the areas that were changed:
1. Tokenization: uchars.index(ch) vs stoi[ch]
2. KV cache allocation strategy: append-growth vs preallocation
3. Loss implementation:
    - Original: explicit softmax + -log p(target)
    - Optimized: log-softmax / logsumexp-based cross entropy (plus further optimization in microgpt_optimized.py)
4. End-to-end training steps: forward + backward + Adam
5. Inference:
    - Original: builds autograd graphs even at inference
    - Optimized: disables graph building and uses float-only softmax sampling

How to run
```bash
python bench_compare.py --steps 200 --infer-samples 100 --reps 3
```
Useful knobs:
- --steps: training steps per run (increase for more stable numbers)
- --infer-samples: number of generated samples per run
- --reps: how many repetitions per benchmark (averaged)
- --block-size, --n-embd, --n-layer, --n-head: control model size

Optional (recommended): enable process RSS deltas by installing psutil:
```bash
pip install psutil
```
Then rerun the benchmark to see RSSΔ changes.

# Interpreting metrics

bench_compare.py prints:
  - Time (s): average wall time per run
  - PyPeak: peak Python allocations measured by tracemalloc (Python heap only)
  - RSSΔ: process RSS delta (if psutil installed)

tracemalloc is best for “Python object churn” (lists, tuples, objects).
RSS is better for “actual process memory”, but is noisier due to allocator behavior and OS caching.

# Summary of observed results (typical)

Your measured runs showed large improvements in the optimized version:
  - Training steps: ~4–5× faster; ~2× lower tracemalloc peak
  - Inference: ~6–7× faster; ~8× lower tracemalloc peak
  - Tokenization: ~1.4× faster; ~4× lower tokenization allocation peak

A notable exception:
	•	Loss microbenchmark sometimes shows a higher tracemalloc peak for the optimized variant, due to Python allocation behavior of large fanout structures. The end-to-end training benchmark still strongly favors the optimized script.

# What changed from microgpt.py to microgpt_optimized.py

This section documents each change, the reasoning, and the math.

*1. Tokenization: uchars.index(ch) → stoi[ch]*

  Original:
  ```code
  [uchars.index(ch) for ch in doc]
  ```
  Optimized:
  ```code
  stoi = {ch: i for i, ch in enumerate(uchars)}
  [stoi[ch] for ch in doc]
  ```
  Reasoning
    - uchars.index(ch) is O(|V|) per character (linear scan).
    - stoi[ch] is O(1) average (hash table).
    - Reduces time and temporary work for every training step.


*2. Autograd node representation: edge-local gradients → _backward closure*

Original Value node stored:
- _children: references
- _local_grads: per-edge derivative constants

Optimized Value node stores:
  - _children
  - a single _backward() closure which implements gradient propagation

Reasoning
  - Storing _local_grads duplicates per-edge storage and increases per-node payload.
  - A closure-based _backward stores gradient logic once per node.
  - Reduces memory overhead and can improve cache locality.

*3. Fused reductions to reduce graph size: dot() and sum_values()*

The original code constructs many intermediate Value nodes for reductions like:
$$\sum_{i=1}^{d} w_i x_i$$

which in Python becomes a chain:
```python
(((w0*x0 + w1*x1) + w2*x2) + ...)
```
Optimized introduces a fused dot(a, b) that:
	•	computes the forward scalar in one loop
	•	stores only the parents once
	•	performs backward in one loop

Dot product math
Forward:
$$y = \sum_{i=1}^{d} a_i b_i$$

Backward:
$$\frac{\partial y}{\partial a_i} = b_i,\quad
\frac{\partial y}{\partial b_i} = a_i$$

So with upstream gradient $$(g = \frac{\partial L}{\partial y})$$:

$$\frac{\partial L}{\partial a_i} \mathrel{+}= b_i \cdot g,\quad
\frac{\partial L}{\partial b_i} \mathrel{+}= a_i \cdot g$$

Reasoning
   - Fewer Python objects (major win in pure Python scalar autograd).
   - Directly supported by end-to-end benchmark improvements.

*4. KV cache handling: preallocated nested cache*

Both scripts implement causal attention via incremental KV caching.

Original
- keys[li].append(k) and values[li].append(v)
- list growth / resizing over time

Optimized
- preallocates keys[layer][t] and values[layer][t] for t in [0..block_size)
- writes in place

Reasoning
- Avoids dynamic list growth and repeated internal resizing.
- More predictable memory behavior under repeated steps.
- A “flat” KV cache was benchmarked but was slower for inference in this codebase due to slice creation; nested was retained.
  
*5. Loss: cross-entropy and log-softmax*

Original: explicit softmax then negative log-likelihood
Softmax: $$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Loss for target class $$(y)$$: $$L = -\log p_y$$

Optimized: log-softmax / logsumexp formulation
Define: $$\mathrm{LSE}(z) = \log\left(\sum_j e^{z_j}\right)$$

Then: $$\log p_y = z_y - \mathrm{LSE}(z)$$

$$L = - (z_y - \mathrm{LSE}(z))$$

Numerical stability
Use the standard max-trick:

$$\mathrm{LSE}(z) = m + \log\left(\sum_j e^{z_j - m}\right),\quad m = \max_j z_j$$

Gradient

$$\frac{\partial L}{\partial z_i} = p_i - \mathbf{1}[i=y]$$

Reasoning
- Avoids materializing all probabilities as long-lived Value objects.
- Provides a stable path for computing loss and gradients.
- In microgpt_optimized.py, logsumexp is implemented as a custom autograd node that computes its gradient using float softmax weights, reducing graph construction overhead.
  
*6. Inference: disable autograd graph construction + float-only softmax sampling*
  
Original inference builds autograd graphs and Value objects for softmax even though gradients are unused.

Optimized inference:
- Sets a global TRACK_GRAD = False
- Computes sampling probabilities as floats:
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
where $$(T)$$ is temperature.

Reasoning
- Eliminates unnecessary node allocations.
- Large speed and memory win (confirmed by inference benchmark).

#Notes on benchmarking and interpretation
- The end-to-end training benchmark is the authoritative measure of overall improvement because it includes real graph construction, backward, and optimizer updates.
- Microbenchmarks (especially loss-only) can be influenced by Python’s allocation patterns (e.g., “many small objects” vs “one large tuple”), which affects tracemalloc peaks.
- In practice, the optimized script delivers strong net improvements in both speed and memory during training and inference.


#Repository layout
- microgpt.py — original implementation
- microgpt_optimized.py — optimized implementation
- bench_compare.py — benchmark harness

#Repro tips

For more stable benchmark results:
- Increase --steps to 500–2000
- Increase --reps to 5
- Ensure no other heavy processes are running
- Use the same Python version across runs
- Install psutil for RSS reporting

Example:

pip install psutil
python bench_compare.py --steps 1000 --infer-samples 200 --reps 5

