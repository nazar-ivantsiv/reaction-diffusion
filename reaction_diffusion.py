import numpy as np
import scipy.signal as sl
import streamlit as st
import matplotlib.pyplot as plt

# init params
iters = 5
d_A = 1.0
d_B = 0.5
f = 0.55
k = 0.62
dt = 1.0

h = 10
w = 10
size = (h, w)

kernel = np.asarray([
    [.05, .2, .05],
    [ .2, -1, .2 ],
    [.05, .2, .05]
])  # 3x3 convolution

np.random.seed(42)

A = np.ones(size) + np.random.rand(*size) * 0.01
B = np.zeros(size) + np.random.rand(*size) * 0.01
B_ones_frac = 0.005  # fraction of B == 1
B_ones_n = int(h + w * B_ones_frac)
mask = (np.random.randint(0, h, B_ones_n), np.random.randint(0, w, B_ones_n))
B[mask] = 1.0

# update initial matrix in a loop
for i in range(iters):
    diffusion_A = d_A * sl.convolve2d(A, kernel, mode='same', boundary='fill', fillvalue=1)
    diffusion_B = d_B * sl.convolve2d(B, kernel, mode='same', boundary='fill')

    print(diffusion_A)

    reaction_A = A * B**2
    reaction_B = A * B**2
    print(reaction_A)

    feed_A = f * (1 - A)
    kill_B = (k + f) * B

    print(feed_A)

    A += (diffusion_A - reaction_A + feed_A) * dt
    B += (diffusion_B + reaction_B - kill_B) * dt

    # plt.subplot(5, 5, i + 1)
    # plt.axis('off')
    # plt.imshow(A + B)

# visualize the heatmap
plt.imshow()
plt.tight_layout()
plt.show()