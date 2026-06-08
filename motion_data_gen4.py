import numpy as np
import scipy.io as sio

# ========================= PARAMETERS =========================
fs = 100e6
fop = 6e9
c = 3e8
num_samples = 1

pos2 = np.array([-300., 0., 0.])
pos3 = np.array([300., 0., 0.])
pos4 = np.array([-100., 300., 0.])
vel2 = np.array([0., 0., 0.])


# ========================= FFT-BASED FRACTIONAL DELAY =========================
def fractional_delay_fft(s, delay_samples):
    """Exact fractional delay using FFT phase shift (circular)"""
    N = len(s)
    k = np.arange(N)
    phase_shift = np.exp(-1j * 2 * np.pi * k * delay_samples / N)
    S = np.fft.fft(s)
    S_shifted = S * phase_shift
    return np.fft.ifft(S_shifted).real if np.isrealobj(s) else np.fft.ifft(S_shifted)


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


# ========================= DATA GENERATION =========================
X1_list = []
X2_list = []
Y = np.zeros((num_samples, 4))
X_ts = []

np.random.seed(42)

for i in range(num_samples):
    pos1 = np.array([0., 100., 0.])
    vx = 50.0
    vy = 10.0
    vel1 = np.array([vx, vy, 0.])

    N = 1_000_000
    x1 = (np.random.rand(N) - 0.5 + 1j * (np.random.rand(N) - 0.5)) * 2.0
    y1 = x1

    y = free_space_propagation(y1, pos1, pos2, vel1, vel2, fs, fop)
    y3 = free_space_propagation(y1, pos1, pos3, vel1, vel2, fs, fop)
    y4 = free_space_propagation(y1, pos1, pos4, vel1, vel2, fs, fop)

    # Block processing (cross-spectrum)
    block_size = 1000
    num_blocks = N // block_size

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

    X1_list.append(y5)
    X2_list.append(y7)
    Y[i] = [pos1[0], pos1[1], vx, vy]
    X_ts.append(1 / fs)

X1 = np.array(X1_list)
X2 = np.array(X2_list)

# ========================= SAVE =========================
dataset = {'X1': X1, 'X2': X2, 'Y': Y, 'X_ts': np.array(X_ts)}
sio.savemat('python_dataset_matched.mat', dataset, do_compression=True)
print("Dataset saved as python_dataset_matched.mat")

# Peak locations
i = 0
row1, col1 = np.unravel_index(np.argmax(X1[i]), X1[i].shape)
row2, col2 = np.unravel_index(np.argmax(X2[i]), X2[i].shape)

print(f"y5 peak -> doppler: {row1}, delay: {col1}")
print(f"y7 peak -> doppler: {row2}, delay: {col2}")