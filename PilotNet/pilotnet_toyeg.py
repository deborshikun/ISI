import numpy as np

# ------------------------------------------------------------------
# Hardcoded toy example inputs for t=1..3 (5×5 matrices)
# ------------------------------------------------------------------
X_t = {
    1: np.array([
        [0, 0, 1, 0, 2],
        [1, 0, 2, 0, 1],
        [1, 0, 2, 2, 0],
        [2, 0, 0, 2, 0],
        [2, 1, 2, 2, 0]
    ], dtype=float),
    2: np.array([
        [0.2, 0.3, 0.9, 0.7, 1.8],
        [1.2, 1.2, 0.8, 0.1, 1.0],
        [0.0, 0.5, 1.3, 1.9, 0.0],
        [1.0, 0.0, 0.1, 2.0, 0.0],
        [2.1, 0.9, 2.0, 2.0, 0.1]
    ], dtype=float),
    3: np.array([
        [0.2, 0.5, 1.3, 0.4, 1.9],
        [1.4, 1.2, 0.8, 0.1, 1.1],
        [0.3, 0.5, 1.3, 1.9, 0.0],
        [1.0, 0.0, 0.1, 2.1, 0.0],
        [2.2, 0.9, 2.0, 2.5, 0.1]
    ], dtype=float)
}

# ------------------------------------------------------------------
# Initial Δ/Σ states for X, Y, Z, P
# ------------------------------------------------------------------
X_last = np.zeros((5, 5))      # rounded X from previous
X_res  = np.zeros((5, 5))      # X-layer residual always zero in toy
Y_res  = np.zeros((5, 5))      # Y-layer accumulator
Z_res  = np.zeros((3, 3))      # Z-layer accumulator
P_res  = 0.0                   # P-layer accumulator

# ------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------
thrX   = 0.5   # threshold for X Δ
thrYZ  = 1.0   # threshold for Y/Z Δ

# ------------------------------------------------------------------
# Convolution kernels
# ------------------------------------------------------------------
W1 = np.array([
    [0,  0, 1],
    [1,  0, 2],
    [1,  0, 2]
], dtype=float)

W2 = np.array([
    [-1, 0, 1],
    [ 0, 0, 1],
    [ 1,-1, 1]
], dtype=float)

W3 = np.array([
    [-1, 0, 1],
    [ 0, 0, 1],
    [ 1,-1, 1]
], dtype=float)

# ------------------------------------------------------------------
# 2D convolution with padding and stride
# ------------------------------------------------------------------
def conv2d(inp, kernel, pad=1, stride=1):
    n, m   = inp.shape
    k      = kernel.shape[0]
    padded = np.pad(inp, pad, mode='constant', constant_values=0.0)
    out_dim = (n + 2*pad - k) // stride + 1
    out     = np.zeros((out_dim, out_dim), dtype=float)
    for i in range(out_dim):
        for j in range(out_dim):
            patch = padded[i*stride:i*stride+k, j*stride:j*stride+k]
            out[i, j] = np.sum(patch * kernel)
    return out

# ------------------------------------------------------------------
# Main loop for t = 1, 2, 3
# ------------------------------------------------------------------
P_deltas = []
P_sums   = []

for t in [1, 2, 3]:
    print(f"\n=== Time step {t} ===")
    X = X_t[t]

        # 1) Round & Δ on X with Δ–Σ (threshold = thrX)
    X_r      = np.floor(X + 0.5)                  # rounded input
    raw_dX   = X_r - X_last                        # raw delta
    dX       = np.where(np.abs(raw_dX) > thrX, raw_dX, 0.0)  # thresholded delta
    X_res    = raw_dX - dX                         # new X-layer residual
    print("X Δ:
", dX)

    # update for next frame
    X_last = X_r.copy()  # update to raw X, not thresholded input

    # 2) Y block: conv → Σ/Δ
    Y_in    = conv2d(dX, W1, pad=1, stride=1)
    Y_sigma = Y_res + Y_in                     # Σ_t
    dY      = Y_sigma - Y_res                  # Δ_t
    dY[np.abs(dY) <= thrYZ] = 0.0              # threshold
    Y_res   = Y_sigma                          # update accumulator
    print("Y Σ:\n", Y_sigma)
    print("Y Δ:\n", dY)

    # 3) Z block: conv → Σ/Δ
    Z_in    = conv2d(dY, W2, pad=1, stride=2)
    Z_sigma = Z_res + Z_in
    dZ      = Z_sigma - Z_res
    dZ[np.abs(dZ) <= thrYZ] = 0.0
    Z_res   = Z_sigma
    print("Z Σ:\n", Z_sigma)
    print("Z Δ:\n", dZ)

    # 4) P scalar: flatten ΔZ, dot W3 → Σ/Δ
    flat    = dZ.flatten()
    w3f     = W3.flatten()
    P_in    = np.dot(w3f, flat)

    P_sigma = P_res + P_in
    dP      = P_sigma - P_res
    if abs(dP) <= thrYZ: dP = 0.0
    P_res   = P_sigma
    print("P Σ:", P_sigma)
    print("P Δ:", dP)

    P_deltas.append(dP)
    P_sums.append(P_sigma)

# Final results
print("\nFinal P Δ (1×3):", P_deltas)
print("Final P Σ (1×3):", P_sums)
