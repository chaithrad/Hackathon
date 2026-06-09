import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
# ========================= PARAMETERS =========================
fs = 100e6
fop = 6e9
c = 3e8
num_samples = 1  # 10000
pos2 = np.array([-300., 0., 0.])
pos3 = np.array([300., 0., 0.])
pos4 = np.array([-100., 300., 0.])
vel2 = np.array([0., 0., 0.])


# %%
# ========================= FFT-BASED FRACTIONAL DELAY =========================
def fractional_delay_fft(s, delay_samples):
    """Exact fractional delay using FFT phase shift (circular)"""
    N = len(s)
    k = np.arange(N)
    phase_shift = np.exp(-1j * 2 * np.pi * k * delay_samples / N)
    S = np.fft.fft(s)
    S_shifted = S * phase_shift
    return np.fft.ifft(S_shifted).real if np.isrealobj(s) else np.fft.ifft(S_shifted)


# %%
def free_space_propagation(s, pos_tx, pos_rx, vel_tx, vel_rx, fs, fop):
    r_vec = pos_rx - pos_tx
    r = np.linalg.norm(r_vec)
    if r < 1e-6:
        r = 1e-6
    tau = r / c
    delay_samples = tau * fs  # can be fractional
    # Radial velocity (MATLAB one-way convention)
    v_radial = np.dot(vel_tx - vel_rx, r_vec) / r
    fd = fop * v_radial / c
    # Apply delay using FFT method (very accurate for this purpose)
    y = fractional_delay_fft(s, delay_samples)
    # Apply Doppler as frequency shift (phase ramp)
    t = np.arange(len(y)) / fs
    doppler_phase = 2 * np.pi * fd * t
    carrier_phase = 2 * np.pi * fop * tau
    y *= np.exp(1j * (doppler_phase - carrier_phase))
    # Free space loss
    y /= r
    return y


# %%
# ========================= DATA GENERATION =========================
X1_list = []
X2_list = []
Ytx = np.zeros((num_samples, 4))
Yrx = np.zeros((num_samples, 8))
X_ts = []
Ds = []
np.random.seed(42)
start_generation_clock = time.time()

for i in range(num_samples):
    # ---- Random Tx position only ----
    x_pos = np.random.uniform(-350, 350)
    y_pos = np.random.uniform(100, 800)
    pos1 = np.array([x_pos, y_pos, 0.])
    # pos1 = np.array([0., 100., 0.])
    vx = 50.0
    vy = 10.0
    vel1 = np.array([vx, vy, 0.])
    N = 10_000_000
    x1 = (np.random.rand(N) - 0.5 + 1j * (np.random.rand(N) - 0.5)) * 2.0
    y1 = x1
    y = free_space_propagation(y1, pos1, pos2, vel1, vel2, fs, fop)
    y3 = free_space_propagation(y1, pos1, pos3, vel1, vel2, fs, fop)
    y4 = free_space_propagation(y1, pos1, pos4, vel1, vel2, fs, fop)

    # Correct block setup
    block_size = 1000
    num_blocks = N // block_size  # should be 10000 if data matches

    z3 = np.zeros((num_blocks, block_size), dtype=complex)
    z5 = np.zeros((num_blocks, block_size), dtype=complex)

    for k in range(num_blocks):
        idx = slice(k * block_size, (k + 1) * block_size)

        Z1 = np.fft.fft(y[idx])
        Z2 = np.fft.fft(y3[idx])
        Z4 = np.fft.fft(y4[idx])

        z3[k, :] = Z1 * np.conj(Z2)
        z5[k, :] = Z4 * np.conj(Z2)

    # 2D FFT for Delay-Doppler map
    y5 = np.abs(np.fft.fft2(z3))
    y7 = np.abs(np.fft.fft2(z5))

    y5 = np.flipud(y5)
    y7 = np.flipud(y7)

    X1_list.append(y5)
    X2_list.append(y7)
    Yrx[i] = [pos2[0], pos2[1], pos3[0], pos3[1], pos4[0], pos4[1], vel2[0], vel2[1]]
    X_ts.append(1 / fs)
    Ds.append(100e3)
    Ytx[i] = [pos1[0], pos1[1], vx, vy]

    # DS 100khz
    # input file: y5,75 2d matrix(10,000X1000)  ,sample time ts, rx pos and rx velo doppler resolution
    # output ifle : tx pos and tx velocity

# print(y5.shape)

X1 = np.array(X1_list)
X2 = np.array(X2_list)

end_generation_clock = time.time()
total_generation_duration = end_generation_clock - start_generation_clock
avg_speef_per_sample = total_generation_duration / num_samples
# %%
# ========================= SAVE =========================
# dataset = {'X1': X1, 'X2': X2, 'Y': Y, 'X_ts': np.array(X_ts)}
datasetIp = {'X1': X1, 'X2': X2, 'Yip': Yrx, 'X_ts': np.array(X_ts), 'Ds': np.array(Ds)}
datasetOp = {'Yop': Ytx}

sio.savemat('python_dataset_Input.mat', datasetIp, do_compression=True)
sio.savemat('python_dataset_Output.mat', datasetOp, do_compression=True)

print("Dataset saved as python_dataset in .mat")

# print("Dataset saved as python_dataset_matched.mat")
# Peak locations
i = 0
row1, col1 = np.unravel_index(np.argmax(X1[i]), X1[i].shape)
row2, col2 = np.unravel_index(np.argmax(X2[i]), X2[i].shape)
print(f"y5 peak -> doppler: {row1}, delay: {col1}")
print(f"y7 peak -> doppler: {row2}, delay: {col2}")

# %%

# ================= VISUALIZATION Part1 =================

# ---- Choose sample index ----
i = 0

# Extract TX position and velocity
tx_x = Ytx[i][0]
tx_y = Ytx[i][1]
vx = Ytx[i][2]
vy = Ytx[i][3]

# RX positions
rx_positions = np.array([pos2, pos3, pos4])

plt.figure(figsize=(7, 7))

# ---- Plot TX ----
plt.scatter(tx_x, tx_y, color='red', s=100, label='TX')

# ---- Plot RX ----
plt.scatter(rx_positions[:, 0], rx_positions[:, 1],
            color='blue', s=100, label='RX')

# ---- Label RX points ----
for idx, (x, y, _) in enumerate(rx_positions):
    plt.text(x + 10, y + 10, f'RX{idx + 1}', fontsize=10)

# ---- Draw velocity vector ----
plt.quiver(tx_x, tx_y, vx, vy,
           color='green', scale=200, label='Velocity')

# ---- Formatting ----
plt.title("TX-RX Geometry")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.show()

# %%
# ================= VISUALIZATION Part2 =================
# ===== FIRST MAP =====
y5_sample = X1[0]

# print(y5_sample[1:20,1:20])
row1, col1 = np.unravel_index(np.argmax(y5_sample), y5_sample.shape)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(20 * np.log10(y5_sample + 1e-6), cmap='jet')
plt.colorbar()
plt.scatter(col1, row1, color='red')
plt.title("DD Map (Rx1 vs Rx2)")

plt.subplot(2, 2, 2)
plt.plot(np.abs(y5_sample[row1, :]))
plt.title("Delay Profile")

plt.subplot(2, 2, 3)
plt.plot(np.abs(y5_sample[:, col1]))
plt.title("Doppler Profile")

'''
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(2,2,4, projection='3d')
Xg, Yg = np.meshgrid(np.arange(1000), np.arange(1000))
ax.plot_surface(Xg, Yg, y5_sample, cmap='jet')
plt.title("3D View")
'''

plt.tight_layout()
plt.show()

# %%
# ================= VISUALIZATION Part3 =================
# ===== SECOND MAP =====
y7_sample = X2[0]
row2, col2 = np.unravel_index(np.argmax(y7_sample), y7_sample.shape)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(20 * np.log10(y7_sample + 1e-6), cmap='jet')
plt.colorbar()
plt.scatter(col2, row2, color='red')
plt.title("DD Map (Rx3 vs Rx2)")

plt.subplot(2, 2, 2)
plt.plot(np.abs(y7_sample[row2, :]))
plt.title("Delay Profile")

plt.subplot(2, 2, 3)
plt.plot(np.abs(y7_sample[:, col2]))
plt.title("Doppler Profile")
'''
ax = plt.subplot(2, 2, 4, projection='3d')
ax.plot_surface(Xg, Yg, y7_sample, cmap='jet')
plt.title("3D View")
'''
plt.tight_layout()
plt.show()