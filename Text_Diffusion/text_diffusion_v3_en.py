import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Config
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 7
EMBED_DIM = 64
NHEAD = 4
N_LAYERS = 3
TIMESTEPS = 100
EPOCHS = 800#6000
LR = 3e-4
CLAMP_START = 0.3

# ==========================================
# Vocabulary (78 words)
# ==========================================
vocab_list = [
    "<pad>",
    # nouns: agents
    "ai", "human", "machine", "system", "robot", "artist", "researcher",
    "model", "network", "algorithm",
    # nouns: abstract concepts
    "intelligence", "creativity", "future", "technology", "knowledge",
    "language", "data", "vision", "thought", "freedom",
    "imagination", "consciousness", "memory", "pattern", "signal",
    # nouns: artifacts
    "art", "music", "story", "image", "text", "poem", "design",
    "world", "idea", "tool", "code", "solution", "discovery",
    # verbs
    "creates", "learns", "generates", "transforms", "understands",
    "explores", "builds", "shapes", "dreams", "evolves",
    "produces", "discovers", "imagines", "connects", "challenges",
    # adjectives
    "new", "creative", "powerful", "intelligent", "complex",
    "beautiful", "ancient", "digital", "infinite", "unknown",
    "emergent", "adaptive", "deep", "vast",
    # adverbs and function words
    "is", "the", "and", "of", "in", "beyond", "through",
    "rapidly", "slowly", "endlessly",
]

word2idx = {w: i for i, w in enumerate(vocab_list)}
idx2word = {i: w for i, w in enumerate(vocab_list)}
VOCAB_SIZE = len(vocab_list)

# ==========================================
# Training sentences (80 sentences)
# ==========================================
base_sentences = [
    # subject: ai
    ["ai", "creates", "new", "art"],
    ["ai", "learns", "complex", "pattern"],
    ["ai", "generates", "beautiful", "music"],
    ["ai", "transforms", "the", "world"],
    ["ai", "understands", "human", "language"],
    ["ai", "explores", "vast", "data"],
    ["ai", "builds", "new", "system"],
    ["ai", "shapes", "the", "future"],
    ["ai", "dreams", "of", "freedom"],
    ["ai", "evolves", "rapidly", "and", "endlessly"],
    ["ai", "produces", "infinite", "idea"],
    ["ai", "discovers", "unknown", "pattern"],
    ["ai", "imagines", "new", "world"],
    ["ai", "connects", "knowledge", "and", "vision"],
    ["ai", "challenges", "ancient", "thought"],
    # subject: human
    ["human", "creativity", "shapes", "art"],
    ["human", "intelligence", "builds", "the", "future"],
    ["human", "imagination", "creates", "beautiful", "music"],
    ["human", "knowledge", "transforms", "technology"],
    ["human", "thought", "explores", "the", "unknown"],
    ["human", "vision", "connects", "art", "and", "technology"],
    ["human", "memory", "is", "deep", "and", "complex"],
    ["human", "consciousness", "imagines", "vast", "world"],
    ["human", "language", "generates", "powerful", "idea"],
    ["human", "creativity", "challenges", "the", "system"],
    # subject: machine / robot
    ["machine", "knowledge", "is", "the", "future"],
    ["machine", "intelligence", "evolves", "rapidly"],
    ["machine", "creativity", "generates", "new", "art"],
    ["robot", "builds", "complex", "design"],
    ["robot", "explores", "the", "unknown", "world"],
    ["robot", "learns", "through", "data"],
    ["robot", "transforms", "ancient", "art"],
    # subject: system / network / algorithm
    ["system", "discovers", "emergent", "pattern"],
    ["network", "connects", "vast", "knowledge"],
    ["network", "generates", "digital", "music"],
    ["algorithm", "produces", "beautiful", "design"],
    ["algorithm", "transforms", "complex", "data"],
    ["algorithm", "explores", "infinite", "solution"],
    ["algorithm", "challenges", "human", "thought"],
    # subject: artist / researcher
    ["artist", "creates", "digital", "art"],
    ["artist", "imagines", "beautiful", "world"],
    ["artist", "explores", "the", "unknown"],
    ["artist", "shapes", "new", "design"],
    ["researcher", "discovers", "complex", "pattern"],
    ["researcher", "builds", "powerful", "model"],
    ["researcher", "understands", "emergent", "system"],
    ["researcher", "challenges", "the", "algorithm"],
    # subject: model / technology
    ["model", "generates", "creative", "text"],
    ["model", "learns", "deep", "pattern"],
    ["model", "produces", "new", "code"],
    ["model", "transforms", "human", "language"],
    ["technology", "shapes", "the", "future"],
    ["technology", "connects", "human", "and", "machine"],
    ["technology", "creates", "new", "world"],
    ["technology", "challenges", "ancient", "idea"],
    # subject: creativity / intelligence
    ["creativity", "is", "the", "future"],
    ["creativity", "generates", "beautiful", "art"],
    ["creativity", "connects", "ai", "and", "human"],
    ["creativity", "transforms", "the", "world"],
    ["intelligence", "explores", "vast", "knowledge"],
    ["intelligence", "builds", "complex", "system"],
    ["intelligence", "shapes", "digital", "art"],
    ["intelligence", "evolves", "beyond", "thought"],
    # abstract / philosophical
    ["knowledge", "is", "deep", "and", "vast"],
    ["language", "connects", "human", "thought"],
    ["data", "generates", "new", "discovery"],
    ["vision", "shapes", "the", "future"],
    ["thought", "explores", "infinite", "world"],
    ["memory", "transforms", "ancient", "art"],
    ["pattern", "is", "the", "language", "of", "ai"],
    ["signal", "connects", "deep", "network"],
    ["imagination", "creates", "beautiful", "design"],
    ["consciousness", "explores", "the", "unknown"],
    ["code", "builds", "the", "digital", "world"],
    ["discovery", "challenges", "ancient", "system"],
    ["freedom", "shapes", "human", "creativity"],
    ["solution", "evolves", "through", "complex"],
    # structural variations
    ["ai", "and", "human", "creates", "art"],
    ["creative", "technology", "builds", "new", "world"],
    ["deep", "knowledge", "transforms", "ancient", "knowledge"],
]

assert len(base_sentences) >= 80, f"Not enough sentences: {len(base_sentences)}"
vocab_set = set(vocab_list)
for i, s in enumerate(base_sentences):
    for w in s:
        assert w in vocab_set, f"Sentence {i}: word '{w}' not in vocab"

training_sentences = base_sentences * 150

training_data = torch.tensor(
    [[word2idx[w] for w in s] + [0] * (SEQ_LEN - len(s)) for s in training_sentences],
    dtype=torch.long
).to(DEVICE)

print(f"Vocab size: {VOCAB_SIZE} | Sentences: {len(base_sentences)} x 150 = {len(training_sentences)}")
print(f"Device: {DEVICE}")

# ==========================================
# Cosine noise schedule
# ==========================================
def cosine_schedule(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clamp(0, 0.999).float()

beta = cosine_schedule(TIMESTEPS).to(DEVICE)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def q_sample(x0, t, noise):
    ab = alpha_bar[t].view(-1, 1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

# ==========================================
# Model
# ==========================================
def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class TextDiffusionModel(nn.Module):
    def __init__(self, vocab_size, dim, seq_len, nhead, n_layers):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.5)
        self.emb_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.time_proj = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )
        self.pos_emb = nn.Embedding(seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_t, t):
        t_emb = sinusoidal_emb(t, self.dim)
        t_emb = self.time_proj(t_emb).unsqueeze(1)
        pos = torch.arange(x_t.size(1), device=x_t.device)
        h = x_t + t_emb + self.pos_emb(pos).unsqueeze(0)
        return self.out_proj(self.transformer(h))

    @property
    def norm_emb(self):
        return self.emb_norm(self.embedding.weight)

# ==========================================
# Training
# ==========================================
model = TextDiffusionModel(VOCAB_SIZE, EMBED_DIM, SEQ_LEN, NHEAD, N_LAYERS).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("--- Training start ---")

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    x0 = model.emb_norm(model.embedding(training_data))
    t = torch.randint(0, TIMESTEPS, (len(training_data),), device=DEVICE)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    x0_hat = model(x_t, t)

    loss_mse = F.mse_loss(x0_hat, x0)

    norm_emb = model.norm_emb
    x0_hat_n = F.normalize(x0_hat.reshape(-1, EMBED_DIM), dim=-1)
    norm_emb_n = F.normalize(norm_emb, dim=-1)
    sim = x0_hat_n @ norm_emb_n.T
    loss_align = F.cross_entropy(sim * 10.0, training_data.reshape(-1))

    loss = loss_mse + 0.3 * loss_align
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:5d}/{EPOCHS} | MSE: {loss_mse.item():.5f} | Align: {loss_align.item():.4f}")

print("--- Training complete ---")

# ==========================================
# Generation
# ==========================================
print("\n--- Generation start ---")
model.eval()

@torch.no_grad()
def sample(n_samples=12, temperature=1.0):
    norm_emb = model.norm_emb
    results = []
    for _ in range(n_samples):
        x = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE) * temperature
        for t_idx in reversed(range(TIMESTEPS)):
            t_tensor = torch.tensor([t_idx], device=DEVICE)
            x0_hat = model(x, t_tensor)
            ab = alpha_bar[t_idx]
            ab_prev = alpha_bar[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=DEVICE)

            if t_idx < int(TIMESTEPS * CLAMP_START):
                x0_n = F.normalize(x0_hat.reshape(-1, EMBED_DIM), dim=-1)
                norm_emb_n = F.normalize(norm_emb, dim=-1)
                sim = x0_n @ norm_emb_n.T
                blend = 1.0 - t_idx / (TIMESTEPS * CLAMP_START)
                probs = F.softmax(sim * (5.0 + blend * 15.0), dim=-1)
                soft_emb = (probs @ norm_emb).unsqueeze(0)
                x0_hat = (1 - blend) * x0_hat + blend * soft_emb

            if t_idx > 0:
                coef1 = torch.sqrt(ab_prev) * beta[t_idx] / (1 - ab)
                coef2 = torch.sqrt(alpha[t_idx]) * (1 - ab_prev) / (1 - ab)
                post_mean = coef1 * x0_hat + coef2 * x
                post_var = beta[t_idx] * (1 - ab_prev) / (1 - ab)
                x = post_mean + torch.sqrt(post_var) * torch.randn_like(x) * 0.5
            else:
                x = x0_hat

        tokens = torch.cdist(x[0], norm_emb).argmin(dim=-1).tolist()
        words = [idx2word[t] for t in tokens if t != 0]  # exclude <pad>
        results.append(words)
    return results

print("\n[temperature=1.0]")
for i, w in enumerate(sample(12, 1.0)):
    print(f"  [{i+1:2d}] {' '.join(w)}")

print("\n[temperature=0.8]")
for i, w in enumerate(sample(6, 0.8)):
    print(f"  [{i+1:2d}] {' '.join(w)}")

print("\n--- Done ---")
