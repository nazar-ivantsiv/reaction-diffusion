import numpy as np
import scipy.signal as sl
import streamlit as st
import matplotlib.pyplot as plt

# init params
iters =99
d_A = 1.0
d_B = 0.5
f = 0.55
k = 0.62
dt = 1.0

h = 200
w = 200
size = (h, w)

kernel = np.asarray([
    [.05, .2, .05],
    [ .2, -1, .2 ],
    [.05, .2, .05]
])  # 3x3 convolution

np.random.seed(42)

A = np.ones(size)# + np.random.rand(*size) * 0.01
B = np.zeros(size)# + np.random.rand(*size) * 0.01
# B_ones_frac = 0.005  # fraction of B == 1
# B_ones_n = int(h + w * B_ones_frac)
# mask = (np.random.randint(0, h, B_ones_n), np.random.randint(0, w, B_ones_n))
# B[mask] = 1.0
h2 = h // 2
w2 = w // 2
_h = max(1, int(h * 0.1))
_w = max(1, int(w * 0.1))

print(h2, _h, w2, _w)

B[h2-_h:h2+_h, w2-_w:w2+_w] = 1
# B[h2-_h + 1:h2+_h - 1, w2-_w + 1:w2+_w - 1] = 0  # fill the inner area with o

print(B)

# update initial matrix in a loop
plt.figure()
for i in range(iters):
    # Laplace ...
    diffusion_A = d_A * sl.convolve2d(A, sundialkernel, mode='same', boundary='wrap')
    diffusion_B = d_B * sl.convolve2d(B, kernel, mode='same', boundary='wrap')
    # diffusion_A = d_A * sl.convolve2d(A, kernel, mode='same', boundary='fill', fillvalue=1)
    # diffusion_B = d_B * sl.convolve2d(B, kernel, mode='same', boundary='fill')

    reaction_A = A * B**2
    reaction_B = A * B**2

    feed_A = f * (1 - A)
    kill_B = (k + f) * B

    A += (diffusion_A - reaction_A + feed_A) * dt
    B += (diffusion_B + reaction_B - kill_B) * dt

    A = np.clip(A, 0, 1)
    B = np.clip(B, 0, 1)

    # u,v = A[1:-1,1:-1], B[1:-1,1:-1]
    # # u = A
    # # v = B
    # Du = d_A
    # Dv = d_B
    # F = f
    # k = k

    # def laplacian_operator(U,V,dx):
    #     Lu = (U[0:-2,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] + U[2:,1:-1] - 4*U[1:-1,1:-1])/dx**2
    #     Lv = (V[0:-2,1:-1] + V[1:-1,0:-2] + V[1:-1,2:] + V[2:,1:-1] - 4*V[1:-1,1:-1])/dx**2
    #     return Lu,Lv

    # Lu,Lv = laplacian_operator(A,B,1)
    # uvv = u*v*v
    # su = Du*Lu - uvv + F *(1-u)
    # sv = Dv*Lv + uvv - (F+k)*v
    # u += dt*su
    # v += dt*sv

    # A = u
    # B = v

    if not i % 10:
        plt.subplot(5, 2, i // 10 + 1)
        plt.axis('off')
        plt.imshow(B)

plt.show()


# visualize the heatmap
plt.imshow(B)
plt.tight_layout()
plt.show()