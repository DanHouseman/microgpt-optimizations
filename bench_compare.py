#!/usr/bin/env python3
"""
Benchmark harness to compare ORIGINAL vs MODIFIED implementations for the areas changed:

Changed areas covered:
  A) Tokenization: uchars.index(ch) vs stoi[ch]
  B) Reduction patterns: chained sum/mul vs fused dot()/sum_values()
  C) Loss: softmax+log vs log-softmax (logsumexp) cross-entropy
  D) KV cache: append-based vs preallocated
  E) Inference: graph-building vs no-grad + float-only softmax

Outputs:
  - Wall time (perf_counter)
  - Peak memory from tracemalloc (Python allocations)
  - (Optional) process RSS peak if psutil is installed

Usage:
  python bench_compare.py --steps 200 --infer-samples 50 --block-size 16 --n-embd 16 --reps 3
"""

import argparse
import gc
import math
import os
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

random.seed(42)


try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    p = psutil.Process(os.getpid())
    return p.memory_info().rss


@dataclass
class BenchResult:
    name: str
    seconds: float
    tracemalloc_peak_bytes: int
    rss_before: Optional[int]
    rss_after: Optional[int]


def run_bench(
    name: str,
    fn: Callable[[], None],
    warmup: int = 1,
    reps: int = 3,
    gc_between_reps: bool = False,
) -> BenchResult:
    for _ in range(warmup):
        fn()
    if gc_between_reps:
        gc.collect()

    tracemalloc.start()
    rss_b = rss_bytes()

    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
        if gc_between_reps:
            gc.collect()
    t1 = time.perf_counter()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_a = rss_bytes()

    return BenchResult(
        name=name,
        seconds=(t1 - t0) / reps,
        tracemalloc_peak_bytes=peak,
        rss_before=rss_b,
        rss_after=rss_a,
    )


# ======================================================================================
# Dataset + vocab (shared)
# ======================================================================================

def load_docs(path: str = "input.txt") -> List[str]:
    if not os.path.exists(path):
        import urllib.request
        names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(names_url, path)
    docs = [line.strip() for line in open(path) if line.strip()]
    random.shuffle(docs)
    return docs


def build_vocab(docs: List[str]) -> Tuple[List[str], int, int, Dict[str, int]]:
    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    stoi = {ch: i for i, ch in enumerate(uchars)}
    return uchars, BOS, vocab_size, stoi


# ======================================================================================
# ORIGINAL Value
# ======================================================================================

class ValueOrig:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, ValueOrig) else ValueOrig(other)
        return ValueOrig(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, ValueOrig) else ValueOrig(other)
        return ValueOrig(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return ValueOrig(self.data ** other, (self,), (other * (self.data ** (other - 1)),))

    def log(self):
        return ValueOrig(math.log(self.data), (self,), (1.0 / self.data,))

    def exp(self):
        e = math.exp(self.data)
        return ValueOrig(e, (self,), (e,))

    def relu(self):
        return ValueOrig(self.data if self.data > 0 else 0.0, (self,), (1.0 if self.data > 0 else 0.0,))

    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1.0)
    def __rtruediv__(self, other): return other * (self ** -1.0)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# ======================================================================================
# MODIFIED Value + fused reductions + grad tracking
# ======================================================================================

TRACK_GRAD = True

class ValueMod:
    __slots__ = ("data", "grad", "_children", "_backward")

    def __init__(self, data, children=(), backward=None):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._backward = backward

    @staticmethod
    def _const(x):
        return x if isinstance(x, ValueMod) else ValueMod(x)

    def __add__(self, other):
        other = ValueMod._const(other)
        out = ValueMod(self.data + other.data, (self, other))
        if TRACK_GRAD:
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward
        return out

    def __mul__(self, other):
        other = ValueMod._const(other)
        out = ValueMod(self.data * other.data, (self, other))
        if TRACK_GRAD:
            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward
        return out

    def __pow__(self, other):
        out = ValueMod(self.data ** other, (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += (other * (self.data ** (other - 1))) * out.grad
            out._backward = _backward
        return out

    def log(self):
        out = ValueMod(math.log(self.data), (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += (1.0 / self.data) * out.grad
            out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = ValueMod(e, (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += e * out.grad
            out._backward = _backward
        return out

    def relu(self):
        out = ValueMod(self.data if self.data > 0 else 0.0, (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
            out._backward = _backward
        return out

    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-ValueMod._const(other))
    def __rsub__(self, other): return ValueMod._const(other) + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (ValueMod._const(other) ** -1.0)
    def __rtruediv__(self, other): return ValueMod._const(other) * (self ** -1.0)

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()


def sum_values(vals: List[ValueMod]) -> ValueMod:
    data = 0.0
    for v in vals:
        data += v.data
    out = ValueMod(data, tuple(vals))
    if TRACK_GRAD:
        def _backward():
            g = out.grad
            for v in vals:
                v.grad += g
        out._backward = _backward
    return out


def dot(a: List[ValueMod], b: List[ValueMod]) -> ValueMod:
    data = 0.0
    for ai, bi in zip(a, b):
        data += ai.data * bi.data
    out = ValueMod(data, tuple(a) + tuple(b))
    if TRACK_GRAD:
        def _backward():
            g = out.grad
            for ai, bi in zip(a, b):
                ai.grad += bi.data * g
                bi.grad += ai.data * g
        out._backward = _backward
    return out


# ======================================================================================
# Shared model builder (original and modified)
# ======================================================================================

def make_matrix(ValueCls, nout: int, nin: int, std: float = 0.08):
    return [[ValueCls(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def linear_orig(x: List[ValueOrig], w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax_orig(logits: List[ValueOrig]):
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm_orig(x: List[ValueOrig]):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def build_original_model(n_layer: int, n_embd: int, block_size: int, n_head: int,
                         vocab_size: int):
    head_dim = n_embd // n_head
    sd = {
        "wte": make_matrix(ValueOrig, vocab_size, n_embd),
        "wpe": make_matrix(ValueOrig, block_size, n_embd),
        "lm_head": make_matrix(ValueOrig, vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f"layer{i}.attn_wq"] = make_matrix(ValueOrig, n_embd, n_embd)
        sd[f"layer{i}.attn_wk"] = make_matrix(ValueOrig, n_embd, n_embd)
        sd[f"layer{i}.attn_wv"] = make_matrix(ValueOrig, n_embd, n_embd)
        sd[f"layer{i}.attn_wo"] = make_matrix(ValueOrig, n_embd, n_embd)
        sd[f"layer{i}.mlp_fc1"] = make_matrix(ValueOrig, 4 * n_embd, n_embd)
        sd[f"layer{i}.mlp_fc2"] = make_matrix(ValueOrig, n_embd, 4 * n_embd)

    params = [p for mat in sd.values() for row in mat for p in row]

    def gpt(token_id: int, pos_id: int, keys, values):
        tok_emb = sd["wte"][token_id]
        pos_emb = sd["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm_orig(x)

        for li in range(n_layer):
            x_res = x
            x = rmsnorm_orig(x)
            q = linear_orig(x, sd[f"layer{li}.attn_wq"])
            k = linear_orig(x, sd[f"layer{li}.attn_wk"])
            v = linear_orig(x, sd[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)

            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs + head_dim]
                k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                    for t in range(len(k_h))
                ]
                attn_w = softmax_orig(attn_logits)
                head_out = [
                    sum(attn_w[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)

            x = linear_orig(x_attn, sd[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_res)]

            x_res = x
            x = rmsnorm_orig(x)
            x = linear_orig(x, sd[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear_orig(x, sd[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_res)]

        return linear_orig(x, sd["lm_head"])

    return sd, params, gpt


# ----- Modified kernels -----

def linear_mod(x: List[ValueMod], w):
    return [dot(wo, x) for wo in w]


def softmax_mod(logits: List[ValueMod]):
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    total = sum_values(exps)
    return [e / total for e in exps]


def rmsnorm_mod(x: List[ValueMod]):
    ms = dot(x, x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def xent_logsoftmax_mod(logits: List[ValueMod], target_id: int) -> ValueMod:
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    lse = sum_values(exps).log() + m
    return -(logits[target_id] - lse)


def alloc_kv_nested(n_layer: int, block_size: int, n_embd: int):
    keys = [[[None] * n_embd for _ in range(block_size)] for _ in range(n_layer)]
    values = [[[None] * n_embd for _ in range(block_size)] for _ in range(n_layer)]
    return keys, values


def alloc_kv_flat(n_layer: int, block_size: int, n_embd: int):
    # flat: keys[layer] is list length block_size*n_embd
    keys = [[None] * (block_size * n_embd) for _ in range(n_layer)]
    values = [[None] * (block_size * n_embd) for _ in range(n_layer)]
    return keys, values


def kv_flat_set(buf2d, layer: int, pos: int, vec: List[ValueMod], n_embd: int):
    base = pos * n_embd
    row = buf2d[layer]
    row[base:base + n_embd] = vec


def kv_flat_get(buf2d, layer: int, pos: int, n_embd: int):
    base = pos * n_embd
    row = buf2d[layer]
    return row[base:base + n_embd]


def build_modified_model(n_layer: int, n_embd: int, block_size: int, n_head: int,
                         vocab_size: int, kv_mode: str = "nested"):
    head_dim = n_embd // n_head
    sd = {
        "wte": make_matrix(ValueMod, vocab_size, n_embd),
        "wpe": make_matrix(ValueMod, block_size, n_embd),
        "lm_head": make_matrix(ValueMod, vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f"layer{i}.attn_wq"] = make_matrix(ValueMod, n_embd, n_embd)
        sd[f"layer{i}.attn_wk"] = make_matrix(ValueMod, n_embd, n_embd)
        sd[f"layer{i}.attn_wv"] = make_matrix(ValueMod, n_embd, n_embd)
        sd[f"layer{i}.attn_wo"] = make_matrix(ValueMod, n_embd, n_embd)
        sd[f"layer{i}.mlp_fc1"] = make_matrix(ValueMod, 4 * n_embd, n_embd)
        sd[f"layer{i}.mlp_fc2"] = make_matrix(ValueMod, n_embd, 4 * n_embd)

    params = [p for mat in sd.values() for row in mat for p in row]

    if kv_mode not in ("nested", "flat"):
        raise ValueError("kv_mode must be 'nested' or 'flat'")

    def gpt(token_id: int, pos_id: int, keys, values):
        tok_emb = sd["wte"][token_id]
        pos_emb = sd["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm_mod(x)

        for li in range(n_layer):
            x_res = x
            x = rmsnorm_mod(x)
            q = linear_mod(x, sd[f"layer{li}.attn_wq"])
            k = linear_mod(x, sd[f"layer{li}.attn_wk"])
            v = linear_mod(x, sd[f"layer{li}.attn_wv"])

            if kv_mode == "nested":
                keys[li][pos_id] = k
                values[li][pos_id] = v
            else:
                kv_flat_set(keys, li, pos_id, k, n_embd)
                kv_flat_set(values, li, pos_id, v, n_embd)

            upto = pos_id + 1
            inv_sqrt = 1.0 / math.sqrt(head_dim)

            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs + head_dim]

                if kv_mode == "nested":
                    k_h = [keys[li][t][hs:hs + head_dim] for t in range(upto)]
                    v_h = [values[li][t][hs:hs + head_dim] for t in range(upto)]
                else:
                    k_h = [kv_flat_get(keys, li, t, n_embd)[hs:hs + head_dim] for t in range(upto)]
                    v_h = [kv_flat_get(values, li, t, n_embd)[hs:hs + head_dim] for t in range(upto)]

                attn_logits = [dot(q_h, k_h[t]) * inv_sqrt for t in range(upto)]
                attn_w = softmax_mod(attn_logits)
                for j in range(head_dim):
                    x_attn.append(sum_values([attn_w[t] * v_h[t][j] for t in range(upto)]))

            x = linear_mod(x_attn, sd[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_res)]

            x_res = x
            x = rmsnorm_mod(x)
            x = linear_mod(x, sd[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear_mod(x, sd[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_res)]

        return linear_mod(x, sd["lm_head"])

    return sd, params, gpt


# ======================================================================================
# Bench targets (patched)
# ======================================================================================

def bench_tokenization(docs: List[str], uchars: List[str], stoi: Dict[str, int], BOS: int,
                       rounds: int = 2000):
    sample_docs = docs[: min(200, len(docs))]

    def orig():
        for i in range(rounds):
            doc = sample_docs[i % len(sample_docs)]
            _ = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

    def mod():
        for i in range(rounds):
            doc = sample_docs[i % len(sample_docs)]
            _ = [BOS] + [stoi[ch] for ch in doc] + [BOS]

    return orig, mod


def bench_kv_alloc(n_layer: int, block_size: int, n_embd: int, rounds: int = 2000):
    # ORIGINAL: append growth
    def orig():
        for _ in range(rounds):
            keys = [[] for _ in range(n_layer)]
            values = [[] for _ in range(n_layer)]
            for t in range(block_size):
                for li in range(n_layer):
                    keys[li].append([0.0] * n_embd)
                    values[li].append([0.0] * n_embd)

    # MOD nested prealloc
    def mod_nested():
        for _ in range(rounds):
            keys, values = alloc_kv_nested(n_layer, block_size, n_embd)
            for t in range(block_size):
                for li in range(n_layer):
                    keys[li][t] = [0.0] * n_embd
                    values[li][t] = [0.0] * n_embd

    # MOD flat prealloc
    def mod_flat():
        for _ in range(rounds):
            keys, values = alloc_kv_flat(n_layer, block_size, n_embd)
            for t in range(block_size):
                for li in range(n_layer):
                    base = t * n_embd
                    keys[li][base:base + n_embd] = [0.0] * n_embd
                    values[li][base:base + n_embd] = [0.0] * n_embd

    return orig, mod_nested, mod_flat


def bench_loss_only(vocab_size: int, rounds: int = 800):
    """
    PATCH: create fresh logits each iteration so we benchmark loss construction/backward,
    not retention quirks from reusing nodes with closure-based backprop.
    """
    target = random.randrange(vocab_size)

    def orig():
        for _ in range(rounds):
            logits = [ValueOrig(random.gauss(0, 1.0)) for _ in range(vocab_size)]
            probs = softmax_orig(logits)
            loss = -probs[target].log()
            loss.backward()

    def mod():
        global TRACK_GRAD
        TRACK_GRAD = True
        for _ in range(rounds):
            logits = [ValueMod(random.gauss(0, 1.0)) for _ in range(vocab_size)]
            loss = xent_logsoftmax_mod(logits, target)
            loss.backward()

    return orig, mod


def bench_train_steps(
    docs: List[str],
    uchars: List[str],
    stoi: Dict[str, int],
    BOS: int,
    vocab_size: int,
    n_layer: int,
    n_embd: int,
    block_size: int,
    n_head: int,
    steps: int,
    kv_mode_mod: str = "nested",
):
    sd_o, params_o, gpt_o = build_original_model(n_layer, n_embd, block_size, n_head, vocab_size)
    sd_m, params_m, gpt_m = build_modified_model(n_layer, n_embd, block_size, n_head, vocab_size, kv_mode=kv_mode_mod)

    def make_adam_state(params):
        return [0.0] * len(params), [0.0] * len(params)

    m_o, v_o = make_adam_state(params_o)
    m_m, v_m = make_adam_state(params_m)

    lr, b1, b2, eps = 0.01, 0.85, 0.99, 1e-8

    def train_step_original(step: int):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos in range(n):
            tid, yid = tokens[pos], tokens[pos + 1]
            logits = gpt_o(tid, pos, keys, values)
            probs = softmax_orig(logits)
            losses.append(-probs[yid].log())
        loss = (1.0 / n) * sum(losses)
        loss.backward()

        lr_t = lr * (1 - step / max(1, steps))
        for i, p in enumerate(params_o):
            g = p.grad
            m_o[i] = b1 * m_o[i] + (1 - b1) * g
            v_o[i] = b2 * v_o[i] + (1 - b2) * (g * g)
            mhat = m_o[i] / (1 - b1 ** (step + 1))
            vhat = v_o[i] / (1 - b2 ** (step + 1))
            p.data -= lr_t * mhat / (math.sqrt(vhat) + eps)
            p.grad = 0.0

    def train_step_modified(step: int):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        if kv_mode_mod == "nested":
            keys, values = alloc_kv_nested(n_layer, block_size, n_embd)
        else:
            keys, values = alloc_kv_flat(n_layer, block_size, n_embd)

        losses = []
        for pos in range(n):
            tid, yid = tokens[pos], tokens[pos + 1]
            logits = gpt_m(tid, pos, keys, values)
            losses.append(xent_logsoftmax_mod(logits, yid))
        loss = sum_values(losses) * (1.0 / n)
        loss.backward()

        lr_t = lr * (1 - step / max(1, steps))
        for i, p in enumerate(params_m):
            g = p.grad
            m_m[i] = b1 * m_m[i] + (1 - b1) * g
            v_m[i] = b2 * v_m[i] + (1 - b2) * (g * g)
            mhat = m_m[i] / (1 - b1 ** (step + 1))
            vhat = v_m[i] / (1 - b2 ** (step + 1))
            p.data -= lr_t * mhat / (math.sqrt(vhat) + eps)
            p.grad = 0.0

    def orig():
        for s in range(steps):
            train_step_original(s)

    def mod():
        global TRACK_GRAD
        TRACK_GRAD = True
        for s in range(steps):
            train_step_modified(s)

    return orig, mod


def bench_inference(
    uchars: List[str],
    stoi: Dict[str, int],
    BOS: int,
    vocab_size: int,
    n_layer: int,
    n_embd: int,
    block_size: int,
    n_head: int,
    samples: int,
    kv_mode_mod: str = "nested",
):
    sd_o, params_o, gpt_o = build_original_model(n_layer, n_embd, block_size, n_head, vocab_size)
    sd_m, params_m, gpt_m = build_modified_model(n_layer, n_embd, block_size, n_head, vocab_size, kv_mode=kv_mode_mod)

    def softmax_float(logits_float: List[float], temperature: float):
        m = max(logits_float)
        exps = [math.exp((x - m) / temperature) for x in logits_float]
        s = sum(exps)
        return [e / s for e in exps]

    temperature = 0.5

    def orig():
        for _ in range(samples):
            keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            token_id = BOS
            for pos in range(block_size):
                logits = gpt_o(token_id, pos, keys, values)
                probs = softmax_orig([l / temperature for l in logits])
                token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
                if token_id == BOS:
                    break

    def mod():
        global TRACK_GRAD
        TRACK_GRAD = False
        for _ in range(samples):
            if kv_mode_mod == "nested":
                keys, values = alloc_kv_nested(n_layer, block_size, n_embd)
            else:
                keys, values = alloc_kv_flat(n_layer, block_size, n_embd)

            token_id = BOS
            for pos in range(block_size):
                logits = gpt_m(token_id, pos, keys, values)
                probs = softmax_float([l.data for l in logits], temperature)
                token_id = random.choices(range(vocab_size), weights=probs)[0]
                if token_id == BOS:
                    break
        TRACK_GRAD = True

    return orig, mod


# ======================================================================================
# Reporting
# ======================================================================================

def fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if x < 1024.0:
            return f"{x:.0f}{unit}"
        x /= 1024.0
    return f"{x:.1f}TB"


def print_results(results: List[BenchResult]):
    def rss_delta(r: BenchResult) -> str:
        if r.rss_before is None or r.rss_after is None:
            return "n/a"
        d = r.rss_after - r.rss_before
        return f"{fmt_bytes(d)}" if d >= 0 else f"-{fmt_bytes(-d)}"

    print("\n=== Benchmark results (per-run averages) ===")
    print(f"{'Benchmark':36} {'Time (s)':>10} {'PyPeak':>10} {'RSSΔ':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r.name:36} {r.seconds:10.4f} {fmt_bytes(r.tracemalloc_peak_bytes):>10} {rss_delta(r):>10}")
    print("-" * 72)

    # Pairwise ratios for ORIG vs MOD in each area where applicable
    pairs = {}
    for r in results:
        parts = r.name.split()
        base = " ".join(parts[:-1])
        kind = parts[-1]
        pairs.setdefault(base, {})[kind] = r

    print("\n=== Ratios (MOD / ORIG) ===")
    print(f"{'Area':36} {'Time':>10} {'PyPeak':>10}")
    print("-" * 60)
    for area, d in pairs.items():
        if "ORIG" in d and "MOD" in d:
            ro, rm = d["ORIG"], d["MOD"]
            time_ratio = rm.seconds / ro.seconds if ro.seconds else float("inf")
            mem_ratio = rm.tracemalloc_peak_bytes / ro.tracemalloc_peak_bytes if ro.tracemalloc_peak_bytes else float("inf")
            print(f"{area:36} {time_ratio:10.3f} {mem_ratio:10.3f}")
    print("-" * 60)

    if psutil is None:
        print("\nNote: psutil not installed; RSSΔ column is n/a. Install with: pip install psutil")


# ======================================================================================
# Main
# ======================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100, help="training steps per benchmark")
    ap.add_argument("--infer-samples", type=int, default=50, help="number of inference samples")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--n-embd", type=int, default=16)
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3, help="repetitions per benchmark")
    ap.add_argument("--gc-between-reps", action="store_true", help="force gc.collect() between reps")
    args = ap.parse_args()

    docs = load_docs()
    uchars, BOS, vocab_size, stoi = build_vocab(docs)
    docs = docs[: min(2000, len(docs))]

    results: List[BenchResult] = []

    # A) Tokenization
    orig_fn, mod_fn = bench_tokenization(docs, uchars, stoi, BOS, rounds=2000)
    results.append(run_bench("Tokenization ORIG", orig_fn, reps=args.reps, gc_between_reps=args.gc_between_reps))
    results.append(run_bench("Tokenization MOD",  mod_fn, reps=args.reps, gc_between_reps=args.gc_between_reps))

    # D) KV allocation: ORIG vs MOD nested vs MOD flat
    orig_fn, mod_nested_fn, mod_flat_fn = bench_kv_alloc(args.n_layer, args.block_size, args.n_embd, rounds=800)
    results.append(run_bench("KV alloc ORIG",        orig_fn,        reps=args.reps, gc_between_reps=args.gc_between_reps))
    results.append(run_bench("KV alloc MOD",         mod_nested_fn,  reps=args.reps, gc_between_reps=args.gc_between_reps))
    results.append(run_bench("KV alloc MOD_FLAT",    mod_flat_fn,    reps=args.reps, gc_between_reps=args.gc_between_reps))

    # C) Loss graph strategy (PATCHED)
    orig_fn, mod_fn = bench_loss_only(vocab_size=vocab_size, rounds=300)
    # loss builds many objects; gc between reps helps stabilize peak attribution
    results.append(run_bench("Loss ORIG", orig_fn, reps=args.reps, gc_between_reps=True))
    results.append(run_bench("Loss MOD",  mod_fn, reps=args.reps, gc_between_reps=True))

    # B+C+D combined: training steps (end-to-end) for nested and flat KV
    orig_fn, mod_fn_nested = bench_train_steps(
        docs, uchars, stoi, BOS, vocab_size,
        args.n_layer, args.n_embd, args.block_size, args.n_head,
        steps=args.steps, kv_mode_mod="nested"
    )
    results.append(run_bench("Train steps ORIG", orig_fn, reps=max(1, args.reps // 1), gc_between_reps=True))
    results.append(run_bench("Train steps MOD",  mod_fn_nested, reps=max(1, args.reps // 1), gc_between_reps=True))

    orig_fn, mod_fn_flat = bench_train_steps(
        docs, uchars, stoi, BOS, vocab_size,
        args.n_layer, args.n_embd, args.block_size, args.n_head,
        steps=args.steps, kv_mode_mod="flat"
    )
    results.append(run_bench("Train steps MOD_FLAT", mod_fn_flat, reps=max(1, args.reps // 1), gc_between_reps=True))

    # E) Inference for nested and flat KV
    orig_inf, mod_inf_nested = bench_inference(
        uchars, stoi, BOS, vocab_size,
        args.n_layer, args.n_embd, args.block_size, args.n_head,
        samples=args.infer_samples, kv_mode_mod="nested"
    )
    results.append(run_bench("Inference ORIG", orig_inf, reps=args.reps, gc_between_reps=args.gc_between_reps))
    results.append(run_bench("Inference MOD",  mod_inf_nested, reps=args.reps, gc_between_reps=args.gc_between_reps))

    orig_inf, mod_inf_flat = bench_inference(
        uchars, stoi, BOS, vocab_size,
        args.n_layer, args.n_embd, args.block_size, args.n_head,
        samples=args.infer_samples, kv_mode_mod="flat"
    )
    results.append(run_bench("Inference MOD_FLAT", mod_inf_flat, reps=args.reps, gc_between_reps=args.gc_between_reps))

    print_results(results)


if __name__ == "__main__":
    main()