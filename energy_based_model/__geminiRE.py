import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# ==========================================
# 1. Energy Model (Simple Quadratic / Hopfield-like)
# ==========================================
class Simple_EnergyModel(nn.Module):
    """
    CNNをやめ、画素同士の相互作用を直接学習するシンプルなエネルギーベースモデル。
    E(x) = - (x^T * W * x + b^T * x)
    ギブズサンプリング（1画素更新）と100%の相性を持つ古典的数理モデルです。
    """
    def __init__(self, dim=784):
        super().__init__()
        # 画素間の相関（ホップフィールドの重み行列 w_ij に相当）
        self.W = nn.Parameter(torch.zeros(dim, dim))
        # 各画素のバイアス（閾値）
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x_flat = x.view(x.size(0), -1)
        
        # エネルギー計算: - (x W x^T + b x)
        # 対称行列にするために自乗の形を計算
        quad = torch.sum(torch.matmul(x_flat, self.W) * x_flat, dim=1)
        linear = torch.matmul(x_flat, self.b)
        
        return -(quad + linear).unsqueeze(1)

# ==========================================
# 2. Pure Gibbs Sampling (MCMC)
# ==========================================
def sample_gibbs_raster(model, x_init, sweeps=5, T=1.0):
    """
    【完全復活：純粋ギブズサンプリング】
    全784画素を順番に巡り、1画素ずつ「0にした時」「1にした時」の
    エネルギー差を計算してサイコロを振ります。
    シンプルなモデルにしたため、1画素の変化に敏感に反応します。
    """
    model.eval()
    x = x_init.clone().detach()
    batch_size, c, h, w = x.shape
    
    with torch.no_grad():
        for sweep in range(sweeps):
            for pos_y in range(h):
                for pos_x in range(w):
                    # --- 1. 対象ピクセルを「0」にした場合の画像とエネルギー ---
                    x_zero = x.clone()
                    x_zero[:, :, pos_y, pos_x] = 0.0
                    e_zero = model(x_zero).squeeze(1)
                    
                    # --- 2. 対象ピクセルを「1」にした場合の画像とエネルギー ---
                    x_one = x.clone()
                    x_one[:, :, pos_y, pos_x] = 1.0
                    e_one = model(x_one).squeeze(1)
                    
                    # --- 3. ボルツマン分布から「1」になる条件付き確率を計算 ---
                    delta_e = e_one - e_zero 
                    p_one = torch.sigmoid(-delta_e / T)
                    
                    # --- 4. 確率p_oneのサイコロを振って、0か1かを決定 ---
                    rand_val = torch.rand(batch_size, device=x.device)
                    decision = (rand_val < p_one).float().unsqueeze(1)
                    x[:, :, pos_y, pos_x] = decision

    model.train()
    return x

# ==========================================
# 3. Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer   = []

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return torch.bernoulli(torch.rand(batch_size, 1, 28, 28))
        n_new   = max(1, batch_size // 4)
        n_old   = batch_size - n_new
        old_imgs = torch.stack(random.sample(self.buffer, n_old))
        new_imgs = torch.bernoulli(torch.rand(n_new, 1, 28, 28))
        return torch.cat([old_imgs, new_imgs], dim=0)

    def add(self, imgs):
        for img in imgs.detach().cpu():
            self.buffer.append(img)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

def plot_images(imgs, title):
    fig = plt.figure(figsize=(5, 5))
    plt.title(title, fontsize=13, pad=10)
    plt.axis('off')
    rows = []
    for i in range(4):
        row = torch.cat([imgs[i * 4 + j][0] for j in range(4)], dim=1)
        rows.append(row)
    grid = torch.cat(rows, dim=0).cpu().numpy()
    plt.imshow(grid, cmap='gray', vmin=0, vmax=1)


# ==========================================
# 4. Main Execution Block (Windows必須対策)
# ==========================================
if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    # データセット準備 ('4' のみ)
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    idx = train_data.targets == 4
    train_data.targets = train_data.targets[idx]
    train_data.data    = train_data.data[idx]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
    print("Data loaded successfully.")

    epochs     = 10
    model      = Simple_EnergyModel() # モデルをシンプルに変更
    buffer     = ReplayBuffer()
    optimizer  = optim.Adam(model.parameters(), lr=1e-3)

    print("Start Training Simple-EBM via Pure Gibbs Sampling...")
    for epoch in range(epochs):
        epoch_cd_loss = 0.0
        
        for i, (x_real, _) in enumerate(train_loader):
            if i >= 20: 
                break
                
            x_real_bin = (x_real > 0.5).float()

            # --- ギブズサンプリング（画像全体を1周スキャン） ---
            x_init = buffer.sample(x_real_bin.size(0))
            x_fake = sample_gibbs_raster(model, x_init, sweeps=1, T=1.0)
            buffer.add(x_fake)

            # --- Contrastive Divergence Loss ---
            e_real = model(x_real_bin)
            e_fake = model(x_fake)
            
            cd_loss  = e_real.mean() - e_fake.mean()
            # L2正則化で重みの暴走を防ぐ
            reg_loss = 0.01 * torch.sum(model.W ** 2)
            total_loss = cd_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_cd_loss += cd_loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1:>2}/{epochs}] | CD Loss (avg): {epoch_cd_loss/20:+.4f}")

    print("Training finished! Generating images...")

    # ==========================================
    # 5. Generation (ノイズから「4」を削り出す)
    # ==========================================
    start_noise = torch.bernoulli(torch.rand(16, 1, 28, 28))

    print("Generating via Long-run Gibbs Sampling (15 sweeps)...")
    # 全画素走査を15周おこない、ノイズを「4」の形へ収束させます
    generated_imgs = sample_gibbs_raster(model, start_noise, sweeps=15, T=1.0)

    plot_images(start_noise,    "Input: Random Noise")
    plot_images(generated_imgs, "Output: Gibbs EBM Generated '4'")
    plt.tight_layout()
    plt.show()