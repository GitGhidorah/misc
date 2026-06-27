"""
Coreference Resolution Demo via Multi-Layer Information Propagation (English version)

Sentence: "There is an apple and an orange here. Let the former be A and the latter be B."
Goal: Establish apple=A and orange=B through a 2-layer simplified Transformer.

Token list:
  0:apple  1:orange  2:.  3:former  4:be  5:A  6:latter  7:be  8:B  9:.

Embedding dimensions (D=8):
  dim0: apple flag
  dim1: orange flag
  dim2: "former" word flag
  dim3: "latter" word flag
  dim4: symbol flag  (A, B)
  dim5: sentence boundary  (.)
  dim6: position  (linear 0.0 ~ 0.9)
  dim7: referential expression flag  (former/latter = 1.0)

Architecture:
  Layer1 Attention : former -> apple (weight=1.0), latter -> orange (weight=1.0)
  Layer1 FFN1      : Key-Value memory -- "former flag detected => write apple score"
  Layer2 Attention : A -> former (weight=1.0),  B -> latter  (weight=1.0)
                     (proximity mask selects the nearest referential expression)
  Layer2 FFN2      : Key-Value memory -- "symbol AND apple info => amplify apple score"

FFN design follows Geva et al. (2021):
  "Transformer Feed-Forward Layers Are Key-Value Memories"
  W1 rows = keys   (what input pattern to detect)
  W2 cols = values (what to output when a neuron fires)
  ReLU    = threshold: only matching neurons fire
"""

import numpy as np
np.set_printoptions(precision=3, suppress=True)

# ── tokens ──────────────────────────────────────────────────
tokens = ["apple", "orange", ".", "former", "be", "A", "latter", "be", "B", "."]
T = len(tokens)
D = 8  # embedding dimension

print("=" * 68)
print("Sentence: \"There is an apple and an orange here.")
print("          Let the former be A and the latter be B.\"")
print()
print("Token sequence:")
for i, t in enumerate(tokens):
    print(f"  pos {i}: {t}")

# ── initial embeddings ───────────────────────────────────────
#          [apple, orange, former, latter, symbol, boundary, pos, referential]
positions = np.arange(T) * 0.1
X0 = np.zeros((T, D))
X0[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, positions[0], 0.0]  # apple
X0[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, positions[1], 0.0]  # orange
X0[2] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, positions[2], 0.0]  # .
X0[3] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, positions[3], 1.0]  # former
X0[4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, positions[4], 0.0]  # be
X0[5] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, positions[5], 0.0]  # A
X0[6] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, positions[6], 1.0]  # latter
X0[7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, positions[7], 0.0]  # be
X0[8] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, positions[8], 0.0]  # B
X0[9] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, positions[9], 0.0]  # .

print("\n" + "=" * 68)
print("【Initial Embeddings X0】")
print("  dims: [apple, orange, former, latter, symbol, boundary, pos, referential]")
for i, t in enumerate(tokens):
    print(f"  {t:8s} (pos {i}): {X0[i]}")

# ── utilities ────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention_with_mask(X, Wq, Wk, Wv, mask=None):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores += mask
    W = softmax(scores)
    return W @ V, W

def ffn_kv(X, W1, b1, W2):
    """
    FFN as Key-Value memory (Geva et al. 2021).
    hidden = ReLU(X @ W1 + b1)   <- pattern matching (keys)
    out    = hidden @ W2          <- value read-out
    """
    hidden = np.maximum(0, X @ W1 + b1)
    out    = hidden @ W2
    return out, hidden

def show_attn(W, label, rows):
    print(f"\n  [{label}]")
    header = 'query \ key'
    print(f"  {header:12s}", end="")
    for t in tokens:
        print(f"{t:9s}", end="")
    print()
    for i in rows:
        t = tokens[i]
        print(f"  {t:12s}", end="")
        for j in range(T):
            w = W[i, j]
            s = f"{w:.3f}"
            if   w > 0.50: s = f"\033[1;32m{w:.3f}\033[0m"
            elif w > 0.10: s = f"\033[33m{w:.3f}\033[0m"
            print(f"{s:9s}", end="")
        print()

# ════════════════════════════════════════════════════════════
# LAYER 1
# Attention : former -> apple  /  latter -> orange
# FFN1      : Key-Value memory writes fruit scores into
#             the hidden states of "former" and "latter"
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("【LAYER 1】")
print("  Attention : former attends to apple  (weight → 1.0)")
print("              latter attends to orange (weight → 1.0)")
print("  FFN1      : Key-Value memory converts the result into")
print("              fruit scores written onto former / latter")

# --- Attention ---
# Q: former uses dim2 (former flag) as query for apple's key
#    latter uses dim3 (latter flag) as query for orange's key
Wq_L1 = np.zeros((D, 2))
Wq_L1[2, 0] = 8.0   # former flag  -> Q[0]  (searches for apple)
Wq_L1[3, 1] = 8.0   # latter flag  -> Q[1]  (searches for orange)

# K: apple  uses dim0 (apple flag)  as key
#    orange uses dim1 (orange flag) as key
Wk_L1 = np.zeros((D, 2))
Wk_L1[0, 0] = 8.0   # apple flag  -> K[0]
Wk_L1[1, 1] = 8.0   # orange flag -> K[1]

# V: pass dim2/dim3 (referential flags) + partial fruit info
Wv_L1 = np.zeros((D, D))
Wv_L1[2, 2] = 1.0   # former flag carried in value
Wv_L1[3, 3] = 1.0   # latter flag carried in value
Wv_L1[0, 0] = 0.5   # partial apple info  (used by FFN1)
Wv_L1[1, 1] = 0.5   # partial orange info (used by FFN1)

out_attn1, W1 = attention_with_mask(X0, Wq_L1, Wk_L1, Wv_L1)
show_attn(W1, "Layer1 Attention weights", rows=[3, 6])  # former, latter

X_res1 = X0 + out_attn1   # residual connection

print("\n  【After Layer1 Attention + Residual — former & latter】")
print("  dims: [apple, orange, former, latter, symbol, boundary, pos, referential]")
for i in [3, 6]:
    print(f"  {tokens[i]:8s}: {X_res1[i]}")

# --- FFN1: Key-Value memory ---
#
# Neuron 0  key   : "dim2 (former flag) is large"
#           value : output 4.0 to dim0 (apple score)
#           => former's hidden state receives apple information
#
# Neuron 1  key   : "dim3 (latter flag) is large"
#           value : output 4.0 to dim1 (orange score)
#           => latter's hidden state receives orange information
#
# Bias = -3.0 acts as a threshold:
#   only tokens with former/latter flag can clear it.

H1 = 4
W1_ffn1 = np.zeros((D, H1))
W1_ffn1[2, 0] = 6.0   # neuron 0 key: dim2 (former flag)
W1_ffn1[0, 0] = 2.0   # neuron 0 key: dim0 (apple info, auxiliary)
W1_ffn1[3, 1] = 6.0   # neuron 1 key: dim3 (latter flag)
W1_ffn1[1, 1] = 2.0   # neuron 1 key: dim1 (orange info, auxiliary)

b1_ffn1 = np.array([-3.0, -3.0, -1.0, -1.0])

W2_ffn1 = np.zeros((H1, D))
W2_ffn1[0, 0] = 4.0   # neuron 0 fires -> output to dim0 (apple score)
W2_ffn1[1, 1] = 4.0   # neuron 1 fires -> output to dim1 (orange score)

ffn1_out, ffn1_hidden = ffn_kv(X_res1, W1_ffn1, b1_ffn1, W2_ffn1)

print("\n  【FFN1 Internal State (Key-Value memory activation)】")
print("  Neuron firing pattern per token:")
print(f"  {'':10s}  neuron0 (former->apple)    neuron1 (latter->orange)")
for i, t in enumerate(tokens):
    h = ffn1_hidden[i]
    if np.any(h > 0.01):
        bar0 = "█" * int(h[0] * 2)
        bar1 = "█" * int(h[1] * 2)
        print(f"  {t:10s}: {h[0]:6.2f} {bar0:10s}     {h[1]:6.2f} {bar1}")
    else:
        print(f"  {t:10s}: (inactive)")

X1 = X_res1 + ffn1_out   # residual connection

print("\n  【After Layer1 (X1) — key tokens】")
for i in [0, 1, 3, 6]:
    t = tokens[i]
    v = X1[i]
    note = ""
    if t == "former": note = f"  <- FFN1 wrote apple score  dim0={v[0]:.2f}"
    if t == "latter": note = f"  <- FFN1 wrote orange score dim1={v[1]:.2f}"
    print(f"  {t:8s}: {v}{note}")

# ════════════════════════════════════════════════════════════
# LAYER 2
# Attention : A -> former  /  B -> latter
#             (proximity mask: attend to the nearest
#              referential expression before self)
# FFN2      : Key-Value memory amplifies
#             "symbol AND apple info"  -> apple score
#             "symbol AND orange info" -> orange score
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("【LAYER 2】")
print("  Attention : A attends to former (weight → 1.0)")
print("              B attends to latter (weight → 1.0)")
print("              (proximity mask ensures nearest referential wins)")
print("  FFN2      : Key-Value memory amplifies")
print("              'symbol AND apple info'  -> large apple score for A")
print("              'symbol AND orange info' -> large orange score for B")

# --- Attention ---
# Symbols A/B (dim4=1.0) query referential expressions (dim7=1.0).
# Proximity mask: score += -(distance * 3.0) for past tokens,
#                          -inf for future tokens.
# This ensures A (pos5) attends to former (pos3, dist=2)
# and B (pos8) attends to latter (pos6, dist=2).

Wq_L2 = np.zeros((D, 2))
Wq_L2[4, 0] = 5.0   # symbol flag -> Q[0]
Wq_L2[6, 1] = 1.0   # position    -> Q[1]

Wk_L2 = np.zeros((D, 2))
Wk_L2[7, 0] = 5.0   # referential flag -> K[0]
Wk_L2[6, 1] = 1.0   # position         -> K[1]

# V: transfer the fruit scores already written into former/latter
Wv_L2 = np.zeros((D, D))
Wv_L2[0, 0] = 1.0   # apple score
Wv_L2[1, 1] = 1.0   # orange score
Wv_L2[4, 4] = 0.5   # keep symbol flag

# proximity mask
dist_mask = np.zeros((T, T))
for i in range(T):
    for j in range(T):
        dist = i - j
        if dist <= 0:
            dist_mask[i, j] = -1e9   # mask future tokens
        else:
            dist_mask[i, j] = -dist * 3.0  # penalise distance

out_attn2, W2 = attention_with_mask(X1, Wq_L2, Wk_L2, Wv_L2, mask=dist_mask)
show_attn(W2, "Layer2 Attention weights", rows=[5, 8])  # A, B

X_res2 = X1 + out_attn2

print("\n  【After Layer2 Attention + Residual — A & B】")
for i in [5, 8]:
    print(f"  {tokens[i]:8s}: {X_res2[i]}")

# --- FFN2: Key-Value memory ---
#
# Neuron 0  key   : "dim0 (apple score) is large  AND  dim4 (symbol flag) is large"
#           value : amplify dim0 (apple score)
#           => fires strongly for A (apple info + symbol flag)
#            but not for B (orange info + symbol flag)
#
# Neuron 1  key   : "dim1 (orange score) is large  AND  dim4 (symbol flag) is large"
#           value : amplify dim1 (orange score)
#           => fires strongly for B, not for A
#
# Bias = -4.0: both conditions must hold to clear the threshold.

H2 = 4
W1_ffn2 = np.zeros((D, H2))
W1_ffn2[4, 0] = 3.0   # neuron 0 key: symbol flag
W1_ffn2[0, 0] = 3.0   # neuron 0 key: apple score  (AND condition)
W1_ffn2[4, 1] = 3.0   # neuron 1 key: symbol flag
W1_ffn2[1, 1] = 3.0   # neuron 1 key: orange score (AND condition)

b1_ffn2 = np.array([-4.0, -4.0, -1.0, -1.0])

W2_ffn2 = np.zeros((H2, D))
W2_ffn2[0, 0] = 5.0   # neuron 0 fires -> amplify apple score
W2_ffn2[1, 1] = 5.0   # neuron 1 fires -> amplify orange score

ffn2_out, ffn2_hidden = ffn_kv(X_res2, W1_ffn2, b1_ffn2, W2_ffn2)

print("\n  【FFN2 Internal State (Key-Value memory activation)】")
print(f"  {'':10s}  neuron0 (symbol+apple->amplify)    neuron1 (symbol+orange->amplify)")
for i, t in enumerate(tokens):
    h = ffn2_hidden[i]
    if np.any(h > 0.01):
        bar0 = "█" * min(int(h[0] * 0.8), 40)
        bar1 = "█" * min(int(h[1] * 0.8), 40)
        print(f"  {t:10s}: {h[0]:7.2f} {bar0:20s}  {h[1]:7.2f} {bar1}")
    else:
        print(f"  {t:10s}: (inactive)")

X2 = X_res2 + ffn2_out

print("\n  【Final Hidden States X2 — key tokens】")
for i in [3, 5, 6, 8]:
    t = tokens[i]
    v = X2[i]
    note = ""
    if t == "A": note = f"  <- apple={v[0]:.2f}, orange={v[1]:.2f}"
    if t == "B": note = f"  <- apple={v[0]:.2f}, orange={v[1]:.2f}"
    print(f"  {t:8s}: {v}{note}")

# ════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("【Final Verification: Coreference Resolution】\n")

checks = [
    ("former", 3, 0, 1, "apple",  "orange"),
    ("latter", 6, 1, 0, "orange", "apple"),
    ("A",      5, 0, 1, "apple",  "orange"),
    ("B",      8, 1, 0, "orange", "apple"),
]

all_ok = True
for name, idx, dim_t, dim_o, target, other in checks:
    v = X2[idx]
    s_t, s_o = v[dim_t], v[dim_o]
    ok = s_t > s_o
    bar_t = "█" * min(int(s_t * 3), 40)
    bar_o = "█" * min(int(s_o * 3), 40)
    status = "✅ OK" if ok else "❌ NG"
    print(f"  {status}  \"{name}\"")
    print(f"         {target:6s} score = {s_t:7.3f}  {bar_t}")
    print(f"         {other:6s} score = {s_o:7.3f}  {bar_o}")
    print()
    if not ok:
        all_ok = False

print("=" * 68)
if all_ok:
    print("✅ All coreference links resolved correctly!")
    print()
    print("   apple  = A   (former -> apple info -> propagated to A)")
    print("   orange = B   (latter -> orange info -> propagated to B)")
else:
    print("❌ Some resolutions failed.")

print("""
【Summary: Multi-Layer Information Propagation】

  Layer 1 Attention:
    former (Q: dim2=1) × apple  (K: dim0=1) -> high score -> former attends to apple
    latter (Q: dim3=1) × orange (K: dim1=1) -> high score -> latter attends to orange
    weight = 1.000 in both cases (complete focus)

  Layer 1 FFN1 (Key-Value memory):
    Neuron 0  key   "former flag is high"  -> fires (activation=4.0)
              value writes apple score (dim0) into former's hidden state
    Neuron 1  key   "latter flag is high"  -> fires (activation=4.0)
              value writes orange score (dim1) into latter's hidden state
    After residual: former dim0 = 16.5,  latter dim1 = 16.5

  Layer 2 Attention (with proximity mask):
    A (pos 5) attends to former (pos 3, dist=2)  -> weight = 1.000
    B (pos 8) attends to latter (pos 6, dist=2)  -> weight = 1.000
    Apple/orange scores already in former/latter are transferred to A/B.

  Layer 2 FFN2 (Key-Value memory, AND-gate pattern):
    Neuron 0  key   "symbol flag AND apple score both large"
              fires for A (48.65),  nearly silent for B (0.15)
              value amplifies apple score in A's hidden state
    Neuron 1  key   "symbol flag AND orange score both large"
              fires for B (48.64),  nearly silent for A
              value amplifies orange score in B's hidden state

  Final hidden states:
    "A"  apple  score = 259.8  >>  orange score = 0.05  =>  apple = A  ✅
    "B"  orange score = 259.8  >>  apple  score = 0.05  =>  orange = B ✅

  Role division:
    Attention -> dynamic routing  (which token to gather info from)
    FFN       -> static knowledge (how to transform the gathered info)
    Residual  -> information accumulation across layers
    Depth     -> multi-step reasoning (2 hops: A->former->apple)
""")
print("=" * 68)
