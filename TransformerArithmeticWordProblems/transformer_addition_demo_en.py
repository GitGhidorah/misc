"""
A conceptual simulation of how a Transformer might process the sentence:

    "Now add two numbers. The first number is 3 + 6 + 4. Add it.
     The second number is 5 + 7 + 9 + 2. Add it. Now add them together."

*** DISCLAIMER ***
This is NOT a real trained Transformer. Real models achieve the behavior
below through learned distributed representations and attention patterns
across billions of parameters, emerging implicitly from training.
Here, each stage (Tokenization -> Attention -> FFN -> Autoregressive
generation) is simplified into an explicit, illustrative simulation of
the *role* each stage plays.
"""

import re
import random
import numpy as np

random.seed(0)
np.random.seed(0)

TEXT = ("Now add two numbers. The first number is three plus six plus four. "
        "Add it. The second number is five plus seven plus nine plus two. "
        "Add it. Now add them together.")


# ============================================================
# Stage 1: Tokenization
# ============================================================
def tokenize(text: str):
    """
    A simplified tokenizer.
    Instead of a real BPE/SentencePiece tokenizer, we split the sentence
    into known words, punctuation, and number words using a naive regex
    (for demonstration purposes only).
    """
    pattern = (
        r"Now add two numbers|Add it|The first number is|The second number is|"
        r"Now add them together|three|six|four|five|seven|nine|two|plus|\.|,"
    )
    tokens = re.findall(pattern, text)
    return tokens


tokens = tokenize(TEXT)

print("=" * 60)
print("Stage 1: Tokenization")
print("=" * 60)
print(tokens)
print(f"Number of tokens: {len(tokens)}\n")


# ============================================================
# Stage 2: Embedding
# ============================================================
DIM = 16  # small embedding dimension for this demo

NUMBER_WORDS = {
    "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "nine": 9, "two": 2,
}


def embed(token: str):
    """
    Hash the token string into a fixed pseudo-embedding vector.
    Number-word tokens get an extra dimension encoding their numeric
    value, plus a flag marking them as "number-like".
    """
    rng = np.random.RandomState(abs(hash(token)) % (2**32))
    vec = rng.normal(size=DIM)
    if token in NUMBER_WORDS:
        # Dimension 0 carries the actual numeric value
        # (in a real model this would be learned, not hand-placed).
        vec[0] = float(NUMBER_WORDS[token])
        vec[1] = 1.0  # a dimension roughly meaning "this is a number token"
    else:
        vec[1] = 0.0
    return vec


embeddings = [embed(t) for t in tokens]

print("=" * 60)
print("Stage 2: Embedding (first 3 tokens shown as an example)")
print("=" * 60)
for t, e in list(zip(tokens, embeddings))[:3]:
    print(f"  '{t}': {np.round(e[:5], 2)} ...({DIM-5} more dims omitted)")
print()


# ============================================================
# Stage 3: Grouping via Attention
# ============================================================
def find_groups(tokens):
    """
    In a real model, attention heads learn to bind each number token
    to the phrase it belongs to ("the first number" / "the second
    number"), often via mechanisms similar to coreference resolution.

    Here we approximate this by treating "." as a sentence boundary and
    grouping the number-word tokens that appear within each segment
    that follows a "The first/second number is" marker.

    -> This stands in for the learned attention heads that would route
       information from "the first/second number" to the relevant
       number tokens later in the sequence.
    """
    groups = []
    current_group = []
    current_label = None

    for t in tokens:
        if t in ("The first number is", "The second number is"):
            current_label = "first" if "first" in t else "second"
        elif t in NUMBER_WORDS:
            current_group.append(t)
        elif t == ".":
            if current_group:
                groups.append((current_label, current_group))
            current_group = []
            current_label = None
    return groups


groups = find_groups(tokens)

print("=" * 60)
print("Stage 3: Grouping via Attention (simplified)")
print("=" * 60)
for label, words in groups:
    print(f"  Label '{label}' <- attended number tokens: {words}")
print()


# ============================================================
# Stage 4: FFN as an approximate addition circuit
# ============================================================
def ffn_addition_circuit(words):
    """
    A simplified re-creation of the addition circuit reported in
    Anthropic's interpretability work (attribution graphs):

      (a) A "magnitude estimation" heuristic: a rough, noisy guess of
          the sum, computed approximately (akin to a log-space estimate).
      (b) A "last-digit lookup" circuit: an exact pattern-match on the
          ones digit (mod 10).
      (c) The two estimates are reconciled by snapping to the nearest
          integer whose last digit matches (b), while staying close
          to the rough estimate from (a).

    Real models likely compute something like this in superposition,
    across many overlapping features; here we make the two estimates
    and their combination fully explicit.
    """
    numbers = [NUMBER_WORDS[w] for w in words]
    exact_sum = sum(numbers)

    # (a) A noisy "rough sense of magnitude"
    magnitude_estimate = exact_sum + np.random.normal(scale=1.5)

    # (b) An exact pattern-match on the last digit (mod 10)
    last_digit_lookup = exact_sum % 10

    # (c) Snap to the closest integer near the estimate whose last
    #     digit matches the exact lookup
    candidates = range(int(magnitude_estimate) - 10, int(magnitude_estimate) + 10)
    best = min(
        candidates,
        key=lambda c: (abs(c - magnitude_estimate) if c % 10 == last_digit_lookup else 1e9),
    )

    return {
        "numbers": numbers,
        "magnitude_estimate": round(magnitude_estimate, 2),
        "last_digit_lookup": last_digit_lookup,
        "snapped_result": best,
        "true_result": exact_sum,
    }


print("=" * 60)
print("Stage 4: FFN approximate-computation simulation")
print("=" * 60)

sub_results = []
for label, words in groups:
    result = ffn_addition_circuit(words)
    sub_results.append(result)
    print(f"  Group '{label}': {' + '.join(words)}")
    print(f"    Magnitude estimate: {result['magnitude_estimate']}")
    print(f"    Last-digit pattern: {result['last_digit_lookup']}")
    print(f"    -> Snapped result: {result['snapped_result']} (true: {result['true_result']})")
print()


# ============================================================
# Stage 5: Autoregressive generation (feeding intermediate results back)
# ============================================================
print("=" * 60)
print("Stage 5: Autoregressive generation (Chain of Thought)")
print("=" * 60)

generated_sequence = list(tokens)  # the original sequence
intermediate_tokens = []

for i, result in enumerate(sub_results):
    new_token = str(result["snapped_result"])
    intermediate_tokens.append(new_token)
    generated_sequence.append(new_token)
    print(f"  Step {i+1}: generated intermediate token '{new_token}' -> appended to sequence")
    print(f"    (from here, the model only needs to attend to this single")
    print(f"     token '{new_token}', instead of the original 3-4 number words)")

# Final step: add the two generated intermediate tokens together
final_numbers = {t: int(t) for t in intermediate_tokens}
exact_final = sum(int(t) for t in intermediate_tokens)
magnitude_estimate = exact_final + np.random.normal(scale=1.5)
last_digit_lookup = exact_final % 10
candidates = range(int(magnitude_estimate) - 10, int(magnitude_estimate) + 10)
final_result = min(
    candidates,
    key=lambda c: (abs(c - magnitude_estimate) if c % 10 == last_digit_lookup else 1e9),
)
generated_sequence.append(str(final_result))

print(f"\n  Final step: computing '{intermediate_tokens[0]}' + '{intermediate_tokens[1]}'")
print(f"    -> generated final token '{final_result}'")

print("\n" + "=" * 60)
print("Tail of the generated sequence")
print("=" * 60)
print(" ".join(generated_sequence[-6:]))
print(f"\n>>> Final output: {final_result}")
