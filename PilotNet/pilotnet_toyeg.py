import numpy as np

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


X_last = np.zeros((5, 5))      
X_res  = np.zeros((5, 5))      
Y_res  = np.zeros((5, 5))      
Z_res  = np.zeros((3, 3))      
P_res  = 0.0                  

thrX   = 0.5   # threshold for X delta
thrYZ  = 1.0   # threshold for Y/Z delta

W1 = np.array([
    [0,  0, 1],
    [1,  0, 2],
    [1,  0, 1]
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

# padding and stride
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

# looping for 3 T frames
P_timesteps = []
P_sigma = 0.0
for t in [1, 2, 3]:
    print(f"\n Time step {t} ")
    X = X_t[t]

    raw_dX   = X.copy()-X_last                       # raw delta
    dX       = np.where(np.abs(raw_dX) >= thrX, raw_dX, 0.0)  # thresholded delta
    X_res    = np.where(np.abs(raw_dX) < thrX, raw_dX, 0.0)   # new X-layer residual
    print("X Δ:", dX)

    # update for next frame
    X_last = X

    Y_in    = conv2d(dX, W1, pad=1, stride=1)
    Y_sigma = Y_res + Y_in                     
    dY      = np.where(np.abs(Y_in) >= thrYZ, Y_in, 0.0)      # thresholded Y delta
    Y_res   = np.where(np.abs(Y_in) < thrYZ, Y_in, 0.0)       # new Y-layer residual
    print("Y Σ:\n", Y_sigma)
    print("Y Δ:\n", dY)

    Z_in    = conv2d(dY, W2, pad=1, stride=2)
    Z_sigma = Z_res + Z_in
    dZ      = np.where(np.abs(Z_in) >= thrYZ, Z_in, 0.0)          # thresholded Z 
    Z_res   = np.where(np.abs(Z_in) < thrYZ, Z_in, 0.0)           # new Z-layer residual
    print("Z Σ:\n", Z_sigma)
    print("Z Δ:\n", dZ)

    flat    = dZ.flatten()
    w3f     = W3.flatten()
    P_in    = np.dot(w3f, flat)

    P_sigma += P_in
    P_timesteps.append(P_in)

# Final
print("\nAll P Time steps:", P_timesteps)
print("Final P Σ :", P_sigma)
