import numpy as np
import scipy.signal as sl
import streamlit as st
import matplotlib.pyplot as plt

# init params
iters = 20
d_A = 1.0
d_B = 0.5
f = 0.55
k = 0.62
dt = 1.0

h = 100
w = 100
size = (h, w)

kernel = np.asarray([
    [.05, .2, .05],
    [ .2, -1, .2 ],
    [.05, .2, .05]
])  # 3x3 convolution

np.random.seed(42)

A = np.ones(size)
B = np.zeros(size)
B_ones_frac = 0.005  # fraction of B == 1
B_ones_n = int(h + w * B_ones_frac)
mask = (np.random.randint(0, h, B_ones_n), np.random.randint(0, w, B_ones_n))
B[mask] = 1.0

# update initial matrix in a loop
for i in range(iters):
    diffusion_A = d_A * sl.convolve2d(A, kernel, mode='same', boundary='fill')
    diffusion_B = d_B * sl.convolve2d(B, kernel, mode='same', boundary='fill')

    reaction_A = A * B**2
    reaction_B = A * B**2

    feed_A = f * (1 - A)
    kill_B = (k + f) * B

    A += (diffusion_A - reaction_A + feed_A) * dt
    B += (diffusion_B + reaction_B - kill_B) * dt

    # plt.subplot(iters // 5, 5, i)
    plt.imshow(A * B)

# visualize the heatmap
# plt.imshow(A * B)
plt.show()