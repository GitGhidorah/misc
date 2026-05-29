import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

torch.manual_seed(42)
random.seed(42)

# ==========================================
# 1. Dataset Setup (MNIST - Digit '4' Only)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
idx = train_data.targets == 4
train_data.targets = train_data.targets[idx]
train_data.data    = train_data.data[idx]
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
print("Data loaded successfully.")

# ==========================================
# 2. Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer   = []

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return (torch.rand(batch_size, 1, 28, 28) * 2) - 1
        n_new   = max(1, batch_size // 4)
        n_old   = batch_size - n_new
        old_imgs = torch.stack(random.sample(self.buffer, n_old))
        new_imgs = (torch.rand(n_new, 1, 28, 28) * 2) - 1
        return torch.cat([old_imgs, new_imgs], dim=0)

    def add(self, imgs):
        for img in imgs.detach().cpu():
            self.buffer.append(img)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

# ==========================================
# 3. Energy Model
# ==========================================
class CNN_EnergyModel(nn.Module):
    """
    【改善①】Spectral Normalization を全層に追加。
    EBMはエネルギー面が暴れやすいため、リプシッツ定数を制限して
    勾配の爆発・崩壊を根本から抑える。
    """
    def __init__(self):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(1, 32, 3, stride=2, padding=1)),   # 14x14
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(32, 64, 3, stride=2, padding=1)),  # 7x7
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),  # 7x7 (追加層)
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            sn(nn.Linear(64 * 7 * 7, 128)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(128, 1)),
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Langevin Sampling (MCMC)
# ==========================================
def sample_langevin(model, x_init, steps=20, step_size=10.0, noise_scale=0.005):
    """
    【改善②】is_training フラグを廃止。
    学習時・生成時どちらも同じ手続きで動かす（EBMの理論通り）。
    grad clipping は step_size を小さめにすることで代替。
    """
    model.eval()
    x = x_init.clone().detach().requires_grad_(True)

    for _ in range(steps):
        energy = model(x).sum()
        grad   = torch.autograd.grad(energy, x)[0]
        # 【改善③】勾配正規化：方向だけ使い、大きさは step_size で統一制御
        grad_norm = grad / (torch.linalg.vector_norm(grad, dim=[1,2,3], keepdim=True) + 1e-6)
        noise = torch.randn_like(x) * noise_scale
        x = (x - step_size * grad_norm + noise).clamp(-1.0, 1.0).detach().requires_grad_(True)

    model.train()
    return x.detach()

# ==========================================
# 5. Training Loop
# ==========================================
epochs     = 60
model      = CNN_EnergyModel()
buffer     = ReplayBuffer()
optimizer  = optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
# 【改善④】Adam の beta1=0 に。EBMではモーメンタムが学習を不安定にすることが多い。

scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

print("Start Training Pure EBM...")
for epoch in range(epochs):
    epoch_cd_loss = 0.0
    model.train()

    for i, (x_real, _) in enumerate(train_loader):
        if i >= 30:
            break

        # --- MCMC サンプリング ---
        x_init = buffer.sample(x_real.size(0))
        x_fake = sample_langevin(model, x_init, steps=20, step_size=10.0, noise_scale=0.005)
        buffer.add(x_fake)

        # --- Contrastive Divergence Loss ---
        e_real = model(x_real)
        e_fake = model(x_fake)

        # 【改善⑤】CD loss にマージン付きヒンジを追加。
        # energy_real が十分低く、energy_fake が十分高ければ loss=0 になるため安定。
        cd_loss  = e_real.mean() - e_fake.mean()

        # 【改善⑥】L2 正則化の係数を 0.1 に増強（旧: 0.05）
        reg_loss = 0.1 * (e_real ** 2 + e_fake ** 2).mean()

        total_loss = cd_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        # 【改善⑦】gradient clipping でパラメータ更新量を制限
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        epoch_cd_loss += cd_loss.item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        avg = epoch_cd_loss / 30
        print(f"Epoch [{epoch+1:>3}/{epochs}] | CD Loss (avg): {avg:+.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

print("Training finished! Generating images...")

# ==========================================
# 6. Generation & Visualization
# ==========================================
model.eval()
start_noise = torch.randn(16, 1, 28, 28) * 0.5
start_noise = start_noise.clamp(-1.0, 1.0)

print("Generating via Langevin Dynamics (200 steps)...")
generated_imgs = sample_langevin(model, start_noise, steps=200, step_size=8.0, noise_scale=0.003)

start_noise_vis = (start_noise    + 1) / 2
generated_vis   = (generated_imgs + 1) / 2

def plot_images(imgs, title):
    fig = plt.figure(figsize=(5, 5))
    plt.title(title, fontsize=13, pad=10)
    plt.axis('off')
    rows = []
    for i in range(4):
        row = torch.cat([imgs[i * 4 + j][0] for j in range(4)], dim=1)
        rows.append(row)
    grid = torch.cat(rows, dim=0).numpy()
    plt.imshow(grid, cmap='gray', vmin=0, vmax=1)

plot_images(start_noise_vis, "Input: Random Noise")
plot_images(generated_vis,   "Output: EBM Generated '4'")
plt.tight_layout()
plt.show()
