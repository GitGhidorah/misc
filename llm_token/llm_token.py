import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math

# 再現性のためのシード固定
torch.manual_seed(42)
random.seed(42)

# ==========================================
# 1. 設定項目
# ==========================================
WORD_GREATER = "大きい"
WORD_LESSER = "小さい"

ELEMENTS = ["A", "B", "C"]

# ハイパーパラメータ
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
NUM_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.002

# ==========================================
# 2. 語彙（ボキャブラリ）の定義
# ==========================================
SPECIAL_TOKENS = ["<PAD>", "<SEP>", "<EOS>"]
VOCAB = SPECIAL_TOKENS + ELEMENTS + [
    WORD_GREATER, WORD_LESSER, "は", "より", "。", "、", 
    "大きい順にならべて", "である"
]

word2idx = {w: i for i, w in enumerate(VOCAB)}
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(VOCAB)

PAD_IDX = word2idx["<PAD>"]
SEP_IDX = word2idx["<SEP>"]
EOS_IDX = word2idx["<EOS>"]

# ==========================================
# 3. データ作成関数（プロンプトと正解ラベルの分離）
# ==========================================
def generate_sample():
    perm = ELEMENTS.copy()
    random.shuffle(perm)
    scores = {elem: 3 - i for i, elem in enumerate(perm)}
    
    pairs = [("A", "B"), ("B", "C"), ("C", "A")]
    prompt = []
    
    for e1, e2 in pairs:
        if random.random() < 0.5:
            first, second = e1, e2
        else:
            first, second = e2, e1
            
        if scores[first] > scores[second]:
            word = WORD_GREATER
        else:
            word = WORD_LESSER
            
        prompt.extend([first, "は", second, "より", word, "。"])
        
    prompt.extend(["大きい順にならべて", "。"])
    
    response = [perm[0], "、", perm[1], "、", perm[2], "。", "である", "。"]
    
    full_tokens = prompt + ["<SEP>"] + response + ["<EOS>"]
    input_ids = [word2idx[w] for w in full_tokens]
    
    # ターゲット（正解ラベル）の作成
    # プロンプト部分は Loss 計算から除外するために -100 に設定する
    labels = [-100] * len(input_ids)
    sep_idx = input_ids.index(SEP_IDX)
    
    # <SEP> の次の位置から先の予測に対してのみ正解ラベルをセット
    labels[sep_idx:] = input_ids[sep_idx+1:] + [EOS_IDX]
    
    return input_ids, labels

class LogicDataset(Dataset):
    def __init__(self, num_samples=3000):
        self.inputs = []
        self.targets = []
        for _ in range(num_samples):
            inp, tgt = generate_sample()
            self.inputs.append(inp)
            self.targets.append(tgt)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx][:-1], dtype=torch.long),
            torch.tensor(self.targets[idx][:-1], dtype=torch.long)
        )

dataset = LogicDataset(3000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
SEQ_LEN = len(dataset[0][0])

# ==========================================
# 4. シンプルかつ確実な Causal GPT モデル
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Causal Attention Matrix
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            h = block(h)
            
        h = self.ln_f(h)
        return self.fc_out(h)

# ==========================================
# 5. モデルの学習
# ==========================================
model = SimpleGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
criterion = nn.CrossEntropyLoss(ignore_index=-100) # プロンプト部分を無視
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print("Transformer (Causal LM) モデルの学習を開始します...")
model.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

# ==========================================
# 6. 推論（テキスト自動生成）関数
# ==========================================
def generate_response(prompt_tokens):
    model.eval()
    input_tokens = prompt_tokens + ["<SEP>"]
    generated_ids = [word2idx[w] for w in input_tokens]
    
    for _ in range(12):
        x = torch.tensor([generated_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            next_token_id = logits[0, -1, :].argmax().item()
            
        if next_token_id == EOS_IDX:
            break
        generated_ids.append(next_token_id)
        
    sep_pos = generated_ids.index(SEP_IDX)
    response_words = [idx2word[i] for i in generated_ids[sep_pos + 1:]]
    return "".join(response_words)

# ==========================================
# 7. 推論テスト実行
# ==========================================
print("\n" + "="*50)
print("--- 推論テスト (テキスト生成) ---")
print("="*50)

test_prompts = [
    ["A", "は", "B", "より", "大きい", "。", "B", "は", "C", "より", "小さい", "。", "C", "は", "A", "より", "小さい", "。", "大きい順にならべて", "。"],
    ["A", "は", "B", "より", "小さい", "。", "B", "は", "C", "より", "大きい", "。", "C", "は", "A", "より", "小さい", "。", "大きい順にならべて", "。"],
    ["A", "は", "B", "より", "小さい", "。", "B", "は", "C", "より", "小さい", "。", "C", "は", "A", "より", "大きい", "。", "大きい順にならべて", "。"]
]

for prompt in test_prompts:
    prompt_text = "".join(prompt)
    output_text = generate_response(prompt)
    print(f"\n入力文: {prompt_text}")
    print(f"出力文: {output_text}")