import torch
import torch.nn as nn
import torch.optim as optim

DIM = 20  # 20-dimensional problem (prevents one-shot shortcut solving)

class LoopedTransformerLayer(nn.Module):
    """
    Standard Transformer Encoder Layer (1 single layer).
    This identical layer is repeatedly reused within the thinking loop.
    """
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        # 1. Multi-Head Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 2. FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class PureLoopedTransformer(nn.Module):
    def __init__(self, problem_dim=DIM, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # Token Embeddings: Map [b (problem)] and [x (current hypothesis)] into d_model dimensions
        self.b_embed = nn.Linear(problem_dim, d_model)
        self.x_embed = nn.Linear(problem_dim, d_model)
        
        # The SINGLE Transformer Layer reused across thinking steps
        self.transformer_layer = LoopedTransformerLayer(d_model=d_model, nhead=4)
        
        # Output Projection: Extract solution update dx from Transformer representation
        self.out_head = nn.Linear(d_model, problem_dim)

    def forward_all_steps(self, b, steps=10):
        # Initial hypothesis: x is initialized as a zero vector
        x_curr = torch.zeros_like(b)
        outputs = []
        
        for _ in range(steps):
            # Form a sequence of 2 tokens: [Token 0: b_embed, Token 1: x_embed]
            tok_b = self.b_embed(b).unsqueeze(1)        # [B, 1, d_model]
            tok_x = self.x_embed(x_curr).unsqueeze(1)    # [B, 1, d_model]
            seq = torch.cat([tok_b, tok_x], dim=1)      # [B, 2, d_model]
            
            # --- Pass through the 1-layer Transformer (Self-Attention interacts b and x) ---
            seq = self.transformer_layer(seq)
            
            # Extract updated residual dx from Token 1 (x token) and update state
            dx = self.out_head(seq[:, 1, :])
            x_curr = x_curr + dx
            outputs.append(x_curr)
            
        return outputs

# ----- Data Generation (Diagonally dominant 20x20 matrix) -----
torch.manual_seed(42)
A_raw = torch.randn(DIM, DIM) * 0.1
A = A_raw + torch.eye(DIM) * 2.0  # Ensure numerical convergence

def generate_hard_system_data(batch_size=2000):
    x_true = torch.randn(batch_size, DIM)
    b = torch.matmul(x_true, A.T)
    return b, x_true

model = PureLoopedTransformer(problem_dim=DIM, d_model=64)
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
criterion = nn.MSELoss()

train_b, train_x_true = generate_hard_system_data(batch_size=3000)

print("--- Training Started (Learning iterative solver using 1-layer Looped Transformer) ---")
MAX_TRAIN_STEPS = 10
for epoch in range(350):
    model.train()
    optimizer.zero_grad()
    
    all_preds = model.forward_all_steps(train_b, steps=MAX_TRAIN_STEPS)
    
    total_loss = 0.0
    for step_idx, pred in enumerate(all_preds):
        weight = (step_idx + 1) / MAX_TRAIN_STEPS
        total_loss += weight * criterion(pred, train_x_true)
        
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 70 == 0:
        print(f"Epoch {epoch+1:3d}/350 - Total Loss: {total_loss.item():.5f}")

print("\n========================================================")
print("--- Evaluation: Step-wise Convergence & Fixed-Point Stability ---")
print("========================================================")
test_b, test_x_true = generate_hard_system_data(batch_size=1000)

model.eval()
with torch.no_grad():
    all_test_preds = model.forward_all_steps(test_b, steps=20)
    
    print(f"{'Loop Count (Thinking Step)':<28} | {'MSE to Solution':<18} | {'Status & Evaluation'}")
    print("-" * 75)
    for loop_count in range(1, 21):
        pred = all_test_preds[loop_count - 1]
        loss = criterion(pred, test_x_true).item()
        
        if loss > 0.3:
            status = "Insufficient Thinking (Unsolved)"
        elif loss > 0.03:
            status = "In Progress (Approaching Solution)"
        elif loss <= 0.03 and loop_count <= MAX_TRAIN_STEPS:
            status = "Solution Reached"
        else:
            status = "Stable Fixed-Point (Robust to Over-thinking)"
            
        if loop_count == MAX_TRAIN_STEPS:
            status += " ★Max Training Steps"
            
        print(f"  Loop {loop_count:<21} | {loss:<18.6f} | {status}")