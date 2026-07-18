import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader, random_split

input_data = sio.loadmat('input_file500_50hDSGUI.mat')
output_data = sio.loadmat('output_file500_50hDSGUI.mat')

X1 = input_data['x1']
X2 = input_data['x2']
Y  = output_data['Y']

print(X1.shape, X2.shape, Y.shape)


def extract_peak(x):
    row, col = np.unravel_index(np.argmax(x), x.shape)
    return row, col

features = []

for i in range(len(X1)):
    r1, c1 = extract_peak(X1[i])
    r2, c2 = extract_peak(X2[i])

    features.append([c1, r1, c2, r2])

features = np.array(features).astype(np.float32)


# Normalize features
features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)

# Normalize outputs
Y = Y.astype(np.float32)

Y[:, 0:2] /= 1000.0   # position
Y[:, 2:4] /= 100.0    # velocity


class PeakDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

dataset = PeakDataset(features, Y)


total_size = len(dataset)

train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size]
)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=4)
test_loader  = DataLoader(test_ds, batch_size=4)


class PeakNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PeakNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

import time
training_start = time.time()

best_val_loss = float('inf')
train_losses = []
val_losses = []

epochs = 200

for epoch in range(epochs):

    start_time = time.time()

    # ---------- TRAIN ----------
    model.train()
    train_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            val_loss += criterion(pred, y_batch).item()

    # ✅ Convert to average
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    # ✅ Store losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    epoch_time = time.time() - start_time

    # ---------- GPU STATS ----------
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    else:
        gpu_mem_alloc = 0
        gpu_mem_reserved = 0

    # ---------- PRINT ----------
    print(f"""
Epoch {epoch+1}
Train Loss: {train_loss:.4f}
Val Loss: {val_loss:.4f}
Time: {epoch_time:.2f}s
GPU Mem Allocated: {gpu_mem_alloc:.2f} MB
GPU Mem Reserved: {gpu_mem_reserved:.2f} MB
LR: {optimizer.param_groups[0]['lr']}
""")

    # ✅ SAVE BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "peaknet_bestGUI.pth")
        print("✅ Best model saved!")


np.save("train_lossesPeakExt.npy", train_losses)
np.save("val_lossesPeakExt.npy", val_losses)
# ================= FINAL SUMMARY =================

training_end = time.time()
total_time = training_end - training_start

print("\n=== TRAINING SUMMARY ===")

print(f"Total Training Time: {total_time:.2f}")
print(f"Final Train Loss: {train_loss:.6f}")
print(f"Final Val Loss: {val_loss:.6f}")
print(f"Total Epochs: {epochs}")

if torch.cuda.is_available():
    gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    print(f"GPU Memory (Allocated): {gpu_mem_alloc:.2f}")
    print(f"GPU Memory (Reserved): {gpu_mem_reserved:.2f}")
else:
    print("GPU Used: CPU")


'''
# ✅ LOAD BEST MODEL
model.load_state_dict(torch.load("peaknet_bestGUI.pth"))
model.to(device)
model.eval()

print("\n==== SAMPLE PREDICTIONS ====")

for i in range(5):
    x_sample, y_true = test_ds[i]

    with torch.no_grad():
        pred = model(x_sample.unsqueeze(0).to(device)).cpu().numpy()[0]

    pred[0:2] *= 1000
    pred[2:4] *= 100

    y_true = y_true.numpy()
    y_true[0:2] *= 1000
    y_true[2:4] *= 100

    print(f"\nSample {i}")
    print("True :", y_true.round(2))
    print("Pred :", pred.round(2))   
   

errors_pos = []
errors_vel = []

model.eval()

with torch.no_grad():
    for x_sample, y_true in test_ds:
    #for x_sample, y_true in val_ds:   # or test_ds if you created

        pred = model(x_sample.unsqueeze(0).to(device)).cpu().numpy()[0]

        pred[0:2] *= 1000
        pred[2:4] *= 100

        y_true = y_true.numpy()
        y_true[0:2] *= 1000
        y_true[2:4] *= 100

        pos_error = np.linalg.norm(pred[0:2] - y_true[0:2])
        vel_error = np.linalg.norm(pred[2:4] - y_true[2:4])

        errors_pos.append(pos_error)
        errors_vel.append(vel_error)

print("\n==== PEAK MODEL PERFORMANCE ====")
print(f"Avg Position Error: {np.mean(errors_pos):.2f} m")
print(f"Avg Velocity Error: {np.mean(errors_vel):.2f} m/s")
'''




