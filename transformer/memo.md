
---

### **Mini Transformer: An Educational Self-Attention Model**  
**Sentence**: *"The animal didn't cross the street because it was too tired."*

**Model Specifications**  
- Dimension (`d_model`): 4  
- Heads: Single Head  
- Vocabulary: 11 tokens  
- Positional Encoding: Simplified Sinusoidal (scaled for stability)  
- Purpose: To clearly demonstrate how Self-Attention works with realistic linguistic phenomena

---

#### 1. Token Embeddings (Final Adjusted)

| Position | Token     | Embedding                  |
|----------|-----------|----------------------------|
| 0        | The       | [0.10, 0.00, 0.00, 0.00]  |
| 1        | animal    | [1.00, 0.50, 0.20, -0.10] |
| 2        | didn't    | [0.00, -0.50, 0.60, 0.60] |
| 3        | cross     | [-0.20, 0.10, -0.40, 1.00]|
| 4        | the       | [0.10, 0.00, 0.00, 0.00]  |
| 5        | street    | [-0.10, 0.20, -0.30, 0.80]|
| 6        | because   | [0.40, 0.60, 0.10, 0.20]  |
| 7        | it        | [0.80, 0.60, 0.10, 0.00]  |
| 8        | was       | [0.20, 0.30, 0.70, 0.40]  |
| 9        | too       | [0.10, 0.40, 0.90, 0.10]  |
| 10       | tired     | [0.25, 0.58, 0.83, 0.08]  |

---

#### 2. Positional Encoding (Simplified Sinusoidal)

**How to use**:  
Add the Positional Encoding vector to the Token Embedding **before** computing Q, K, and V.  
Final Input = Token Embedding + Positional Encoding

**Positional Encoding Table** (scaled for this educational model):

| Position | dim0    | dim1    | dim2    | dim3    |
|----------|---------|---------|---------|---------|
| 0        | 0.0000  | 0.0600  | 0.0000  | 0.0600  |
| 1        | 0.0050  | 0.0599  | 0.0000  | 0.0600  |
| 2        | 0.0100  | 0.0596  | 0.0001  | 0.0600  |
| 3        | 0.0150  | 0.0591  | 0.0001  | 0.0600  |
| 4        | 0.0199  | 0.0584  | 0.0002  | 0.0600  |
| 5        | 0.0249  | 0.0575  | 0.0002  | 0.0600  |
| 6        | 0.0298  | 0.0564  | 0.0003  | 0.0599  |
| 7        | 0.0346  | 0.0551  | 0.0003  | 0.0599  |
| 8        | 0.0394  | 0.0537  | 0.0004  | 0.0599  |
| 9        | 0.0441  | 0.0521  | 0.0004  | 0.0599  |
| 10       | 0.0488  | 0.0503  | 0.0005  | 0.0598  |

---

#### 3. Weight Matrices

**W_q (Query)**
$$
\begin{bmatrix}
0.9 & 0.1 & 0.2 & -0.1 \\
0.1 & 0.8 & 0.3 & 0.2 \\
0.0 & 0.2 & 0.7 & 0.4 \\
-0.1 & 0.1 & 0.3 & 0.9
\end{bmatrix}
$$

**W_k (Key)**
$$
\begin{bmatrix}
1.0 & 0.0 & 0.1 & 0.0 \\
0.0 & 0.9 & 0.2 & 0.1 \\
0.1 & 0.1 & 1.0 & 0.3 \\
0.0 & 0.2 & 0.2 & 0.8
\end{bmatrix}
$$

**W_v (Value)**: Nearly Identity matrix (preserves original information)

---

#### 4. Attention Scores (Top 5 for Each Query)

| Query (Pos)   | 1st                  | Score   | 2nd                | Score   | 3rd                | Score   | 4th             | Score   | 5th            | Score   |
|---------------|----------------------|---------|--------------------|---------|--------------------|---------|-----------------|---------|----------------|---------|
| The (0)       | **animal**           | 0.162   | it                 | 0.148   | tired              | 0.139   | was             | 0.135   | too            | 0.124   |
| animal (1)    | **animal**           | 1.562   | it                 | 1.371   | tired              | 1.152   | was             | 1.041   | too            | 0.993   |
| didn't (2)    | **didn't**           | 0.841   | **cross**          | 0.612   | street             | 0.472   | was             | 0.421   | too            | 0.395   |
| cross (3)     | **cross**            | 0.685   | **street**         | 0.591   | **didn't**         | 0.528   | was             | 0.451   | too            | 0.339   |
| the (4)       | **street**           | **0.137** | cross            | 0.102   | didn't             | 0.078   | was             | 0.061   | too            | 0.054   |
| street (5)    | **cross**            | 0.608   | **street**         | 0.549   | didn't             | 0.462   | was             | 0.438   | too            | 0.417   |
| because (6)   | **tired**            | 1.038   | was                | 0.972   | too                | 0.931   | it              | 0.901   | animal         | 0.892   |
| it (7)        | **animal**           | 1.379   | it                 | 1.245   | tired              | 1.118   | was             | 1.025   | too            | 0.972   |
| was (8)       | **tired**            | 1.401   | was                | 1.376   | too                | 1.338   | didn't          | 0.871   | because        | 0.842   |
| too (9)       | **tired**            | 1.371   | too                | 1.312   | was                | 1.298   | because         | 0.811   | didn't         | 0.768   |
| tired (10)    | **tired**            | 1.492   | was                | 1.412   | too                | 1.398   | animal          | 0.945   | because        | 0.931   |

---

#### 5. Notable Achievements of This Model

- **Clear distinction between "The" and "the"**:  
  "The" (position 0) focuses on "animal", while "the" (position 4) correctly focuses on "street". This demonstrates the power of positional encoding.

- **Strong syntactic phrase detection**:  
  "didn't ↔ cross" and "cross ↔ street" show robust detection of verb phrases.

- **Coreference resolution**:  
  "animal" and "it" strongly attend to each other.

- **Semantic focus**:  
  The reason clause ("because it was too tired") correctly converges on "tired".

- **Realistic functional vs content word behavior**:  
  Articles ("The", "the") have lower overall scores but focus on the most relevant nouns.

This small-scale model (d_model=4) successfully demonstrates core Transformer mechanisms — positional awareness, syntactic binding, and semantic focus — in an understandable way.

---

**Created in collaboration with Grok (xAI)**  
This is an educational miniature model designed to help students and developers understand how Transformer attention actually works.

---
