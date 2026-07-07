import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import numpy as np

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Prepare Image Data (MNIST)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Use a larger batch size for CPU acceleration to reduce total step count
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

# ==========================================
# 2. Define Multimodal Networks (3D Extended)
# ==========================================

# Image Encoder (CNN)
class ImageEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 3) # === Extended to 3D Embedding Space ===
        )
    def forward(self, x):
        return self.conv(x)

# Text/Label Encoder (MLP)
class TextEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # === Extended to 3D Embedding Space ===
        )
    def forward(self, x):
        return self.net(x)

# Predictor (Predicts target modality embedding)
class Predictor3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3) # === 3D Compatible ===
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Loss Functions (SIGReg Core Concept)
# ==========================================
def variance_loss(z, eps=1e-4):
    # Constrain the standard deviation of each dimension to approach 1 (Prevents Dimensional Collapse)
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(torch.relu(1.0 - std))

def covariance_loss(z):
    # Decorrelate dimensions to prevent all information from flattening into a single line
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    # Target off-diagonal elements (covariances) to approach 0
    diag_mask = torch.eye(d, device=z.device)
    loss = (cov * (1 - diag_mask)).pow(2).sum() / d
    return loss

# ==========================================
# 4. Training Loop
# ==========================================
img_enc = ImageEncoder3D()
txt_enc = TextEncoder3D()
pred_i2t = Predictor3D()
pred_t2i = Predictor3D()

params = list(img_enc.parameters()) + list(txt_enc.parameters()) + \
         list(pred_i2t.parameters()) + list(pred_t2i.parameters())
optimizer = optim.Adam(params, lr=0.002)

epochs = 10
print("Starting 3D training on CPU (Will complete in ~2-3 minutes)...")

for epoch in range(epochs):
    img_enc.train(); txt_enc.train()
    total_loss = 0
    
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        
        # Convert labels to One-hot vectors (simple text feature proxy) and inject mild noise
        txts = torch.nn.functional.one_hot(labels, num_classes=10).float()
        txts += torch.randn_like(txts) * 0.1
        
        # Extract embeddings
        z_i = img_enc(imgs)
        z_t = txt_enc(txts)
        
        # --- LeVLJEPA Core: Asymmetric Prediction + Stop-Gradient on Targets ---
        # 1. Predict Text from Image (Freeze Text Encoder via detach)
        p_t = pred_i2t(z_i)
        loss_i2t = torch.mean((p_t - z_t.detach()) ** 2)
        
        # 2. Predict Image from Text (Freeze Image Encoder via detach)
        p_i = pred_t2i(z_t)
        loss_t2i = torch.mean((p_i - z_i.detach()) ** 2)
        
        # 3. Regularization Term (SIGReg Concept: Spreading representations uniformly in 3D)
        reg_loss = (variance_loss(z_i) + variance_loss(z_t)) * 1.0 + \
                   (covariance_loss(z_i) + covariance_loss(z_t)) * 1.0
        
        loss = loss_i2t + loss_t2i + reg_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 5. Visualize 3D Embedding Space
# ==========================================
img_enc.eval()
print("\n[Done] Rendering 3D plot...")
print("-> Note: You can left-click and drag the plot window to rotate and inspect it from any angle.")

with torch.no_grad():
    # Fetch a subset of data for validation plotting
    test_imgs, test_labels = next(iter(train_loader))
    z_i_test = img_enc(test_imgs).numpy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Render 3D scatter plot
scatter = ax.scatter(z_i_test[:, 0], z_i_test[:, 1], z_i_test[:, 2], 
                     c=test_labels.numpy(), cmap='tab10', alpha=0.8, s=20)

# Configure colorbar and axis labels
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, ticks=range(10))
cbar.set_label('MNIST Digit Labels')

ax.set_title("LeVLJEPA 3D Embedding Space (Image Encoder Output)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
ax.grid(True, alpha=0.3)

plt.show()