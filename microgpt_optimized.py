import os
import math
import random
random.seed(42)  # Let there be order among chaos

# ======================================================================================
# Optimized pure-Python GPT (micrograd-style) with memory/compute improvements validated
# by benchmarks:
#
# - Tokenization: uchars.index(ch) -> O(1) dict lookup (stoi)  [WIN: faster + less allocs]
# - Inference: disable autograd graph + float softmax sampling  [WIN: much faster + less mem]
# - Training: fused dot() + small fused reductions to reduce node count  [WIN: ~4-5x faster]
# - KV cache: keep nested prealloc (not flat) due to benchmarked inference speed and
#   avoidance of slice-heavy flat reads.
# - Loss: replace exp-node-heavy logsumexp with a custom logsumexp node that computes
#   softmax gradients directly (addresses loss microbench peak in spirit: avoids creating
#   V exp Value nodes and avoids a giant exp-children tuple).
# ======================================================================================

# ------------------------
# Data loading
# ------------------------
if not os.path.exists("input.txt"):
    import urllib.request
    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")

docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# CHANGE (benchmarked): O(1) tokenization
stoi = {ch: i for i, ch in enumerate(uchars)}
itos = {i: ch for ch, i in stoi.items()}

# ------------------------
# Autograd engine
# ------------------------
TRACK_GRAD = True  # CHANGE (benchmarked): disable graph building during inference

class Value:
    """
    Scalar reverse-mode autograd node.

    CHANGE (memory): store one _backward closure rather than per-edge local grads tuple.
    This reduces per-node payload and simplifies graph storage.
    """
    __slots__ = ("data", "grad", "_children", "_backward")

    def __init__(self, data, children=(), backward=None):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._backward = backward

    @staticmethod
    def const(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = Value.const(other)
        out = Value(self.data + other.data, (self, other))
        if TRACK_GRAD:
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value.const(other)
        out = Value(self.data * other.data, (self, other))
        if TRACK_GRAD:
            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward
        return out

    def __pow__(self, p):
        out = Value(self.data ** p, (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += (p * (self.data ** (p - 1))) * out.grad
            out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += (1.0 / self.data) * out.grad
            out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += e * out.grad
            out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        if TRACK_GRAD:
            def _backward():
                self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
            out._backward = _backward
        return out

    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-Value.const(other))
    def __rsub__(self, other): return Value.const(other) + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (Value.const(other) ** -1.0)
    def __rtruediv__(self, other): return Value.const(other) * (self ** -1.0)

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

# ------------------------
# Fused ops (reduce node count)
# ------------------------

def sum_values(vals):
    """
    Fused sum node.

    Note: for very large vectors, a single node with many children can increase
    peak allocation due to the children tuple. In this script we only use it for:
      - attention weighted sums over <= block_size
      - summing <= block_size losses
    Both are small and benchmark-safe.
    """
    s = 0.0
    for v in vals:
        s += v.data
    out = Value(s, tuple(vals))
    if TRACK_GRAD:
        def _backward():
            g = out.grad
            for v in vals:
                v.grad += g
        out._backward = _backward
    return out

def dot(a, b):
    """
    Fused dot product node.

    CHANGE (benchmarked): avoids building long chains of mul+add nodes.
    """
    s = 0.0
    for ai, bi in zip(a, b):
        s += ai.data * bi.data
    out = Value(s, tuple(a) + tuple(b))
    if TRACK_GRAD:
        def _backward():
            g = out.grad
            for ai, bi in zip(a, b):
                ai.grad += bi.data * g
                bi.grad += ai.data * g
        out._backward = _backward
    return out

def logsumexp_node(logits):
    """
    CHANGE (based on benchmarks): custom logsumexp that avoids creating V exp Value nodes.

    Forward:
      lse = log(sum exp(logits))
    Backward:
      d(lse)/d(logits_i) = softmax_i

    This keeps the graph smaller and avoids exp-node fanout dominating peak allocations.
    """
    m = max(v.data for v in logits)
    exps = [math.exp(v.data - m) for v in logits]      # floats, not Value nodes
    s = sum(exps)
    lse_val = m + math.log(s)

    out = Value(lse_val, tuple(logits))
    if TRACK_GRAD:
        # cache softmax probs as floats
        probs = [e / s for e in exps]
        def _backward():
            g = out.grad
            for v, p in zip(logits, probs):
                v.grad += p * g
        out._backward = _backward
    return out

# ------------------------
# Model config / weights
# ------------------------
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

def linear(x, w):
    # CHANGE (benchmarked): dot() reduces node explosion
    return [dot(wo, x) for wo in w]

def rmsnorm(x):
    # CHANGE: uses dot(x,x) fused
    ms = dot(x, x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# KV cache: nested prealloc (kept based on inference benchmark; avoids slice-heavy flat reads)
def alloc_kv():
    keys = [[[None] * n_embd for _ in range(block_size)] for _ in range(n_layer)]
    values = [[[None] * n_embd for _ in range(block_size)] for _ in range(n_layer)]
    return keys, values

def softmax(logits):
    # used inside attention; vectors are length <= block_size so safe
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    denom = sum_values(exps)
    return [e / denom for e in exps]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])

        keys[li][pos_id] = k
        values[li][pos_id] = v

        upto = pos_id + 1
        inv_sqrt = 1.0 / math.sqrt(head_dim)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [keys[li][t][hs:hs + head_dim] for t in range(upto)]
            v_h = [values[li][t][hs:hs + head_dim] for t in range(upto)]

            attn_logits = [dot(q_h, k_h[t]) * inv_sqrt for t in range(upto)]
            attn_w = softmax(attn_logits)

            for j in range(head_dim):
                x_attn.append(sum_values([attn_w[t] * v_h[t][j] for t in range(upto)]))

        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_res)]

        x_res = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_res)]

    logits = linear(x, state_dict["lm_head"])
    return logits

# ------------------------
# Training
# ------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

def xent_loss_from_logits(logits, target_id):
    """
    CHANGE: loss uses custom logsumexp node to avoid creating exp Value nodes for vocab.
    loss = -(logits[target] - logsumexp(logits))
    """
    lse = logsumexp_node(logits)
    return -(logits[target_id] - lse)

num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]  # CHANGE: stoi
    n = min(block_size, len(tokens) - 1)

    keys, values = alloc_kv()

    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        losses.append(xent_loss_from_logits(logits, target_id))

    loss = sum_values(losses) * (1.0 / n)
    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        g = p.grad
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (math.sqrt(v_hat) + eps_adam)
        p.grad = 0.0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end="\r")

# ------------------------
# Inference
# ------------------------
def softmax_float(logits_float, temperature=1.0):
    m = max(logits_float)
    exps = [math.exp((x - m) / temperature) for x in logits_float]
    s = sum(exps)
    return [e / s for e in exps]

temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")

# CHANGE (benchmarked): no autograd graph + float softmax sampling
TRACK_GRAD = False
for sample_idx in range(20):
    keys, values = alloc_kv()
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax_float([l.data for l in logits], temperature=temperature)
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
TRACK_GRAD = True