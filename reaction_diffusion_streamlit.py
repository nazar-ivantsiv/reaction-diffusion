import time
import numpy as np
import pandas as pd
import scipy.signal as sl
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import streamlit as st
import SessionState  # Assuming SessionState.py lives on this folder


def main():
    session = SessionState.get(run_id=0)

    if st.sidebar.button("Reset"):
        session.run_id += 1

    # Init params
    random_noise = st.sidebar.checkbox('random_noise', value=False, key=session.run_id)
    draw_square = st.sidebar.checkbox('draw_square', value=True, key=session.run_id)
    iters = st.sidebar.slider('iterations', min_value=10, max_value=10000, step=10, value=6000, key=session.run_id)
    d_A = st.sidebar.slider('d_A', min_value=0.0, max_value=2.0, step=0.1, value=1.0, key=session.run_id)
    d_B = st.sidebar.slider('d_B', min_value=0.0, max_value=2.0, step=0.1, value=0.5, key=session.run_id)
    f = st.sidebar.slider('feed_rate', min_value=0.0, max_value=0.5, step=0.005, value=0.055, key=session.run_id)
    k = st.sidebar.slider('kill_rate', min_value=0.0, max_value=0.5, step=0.005, value=0.062, key=session.run_id)
    # dt = st.sidebar.slider('dt', min_value=0.1, max_value=1.0, step=0.1, value=1.0, key=session.run_id)
    dt = 1.0
    side = st.sidebar.slider('side', min_value=10, max_value=1000, step=1, value=200, key=session.run_id)

    h = side
    w = side
    size = (h, w)

    kernel = np.asarray([
        [.05, .2, .05],
        [ .2, -1, .2 ],
        [.05, .2, .05]
    ])

    A = np.ones(size)
    B = np.zeros(size)

    # Add randomness to the pattern
    if random_noise:
        rng_seed = st.sidebar.slider('rng_seed', min_value=0, max_value=1000, step=10, value=40, key=session.run_id)
        rng_noise_amount = st.sidebar.slider('rng_noise_amount', min_value=0.0, max_value=0.5, step=0.005, value=0.005, key=session.run_id)
        np.random.seed(rng_seed)

        B_ones_frac = rng_noise_amount  # fraction of B == 1
        B_ones_n = int(h + w * B_ones_frac)
        mask = (np.random.randint(0, h, B_ones_n), np.random.randint(0, w, B_ones_n))
        B[mask] = 1.0

    # Add a square in the center of the image
    if draw_square:
        fraction = 0.1
        h2 = h // 2
        w2 = w // 2
        _h = max(1, int(h * fraction))
        _w = max(1, int(w * fraction))
        B[h2-_h:h2+_h, w2-_w:w2+_w] = 1
        B[h2-_h + 1:h2+_h - 1, w2-_w + 1:w2+_w - 1] = 0  # fill the inner area with o

    st.sidebar.write(f'Current parameters:')
    st.sidebar.dataframe(pd.DataFrame(dict(iters=iters, d_A=d_A, d_B=d_B), index=[0]))
    st.sidebar.dataframe(pd.DataFrame(dict(feed=f, kill=k, dt=dt), index=[0]))

    # Solve the equation in a loop
    t0 = time.perf_counter()
    fig1 = plt.figure()

    # A, B, interim_b = solve(A, B, kernel, d_A, d_B, f, k, dt, iters)
    A, B, interim_b = solve_torch(A, B, kernel, d_A, d_B, f, k, dt, iters)

    # Visualize interim results
    for i, b in enumerate(interim_b):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(b, cmap=plt.cm.PRGn)


    # Visualize the final result heatmap
    fig2 = plt.figure()
    plt.axis('off')
    plt.imshow(B, cmap=plt.cm.PRGn)
    plt.colorbar()
    st.pyplot(fig2)

    # Visualize interm results
    # plt.tight_layout()
    st.pyplot(fig1)

    st.write(f'Elapsed: {time.perf_counter() - t0:.2f} sec')


def solve(A, B, kernel, d_A, d_B, f, k, dt, iters):
    interim_b = []
    for i in range(iters):
        diffusion_A = d_A * sl.convolve2d(A, kernel, mode='same', boundary='wrap')  # 2D Laplacian convolution
        diffusion_B = d_B * sl.convolve2d(B, kernel, mode='same', boundary='wrap')

        reaction_A = A * B**2
        reaction_B = A * B**2

        feed_A = f * (1 - A)
        kill_B = (k + f) * B

        A += (diffusion_A - reaction_A + feed_A) * dt
        B += (diffusion_B + reaction_B - kill_B) * dt

        # Save interim results
        n = iters // 10
        if not i % n:
            interim_b.append(B)

    return A, B, interim_b


def solve_torch(A, B, kernel, d_A, d_B, f, k, dt, iters):
    A_t = torch.tensor(A)
    B_t = torch.tensor(B)
    kernel_t = torch.tensor(kernel)
    A_t = A_t.expand(1, 1, -1, -1)
    B_t = B_t.expand(1, 1, -1, -1)
    kernel_t = kernel_t.expand(1, 1, -1, -1)

    interim_b = []
    for i in range(iters):
        diffusion_A = d_A * F.conv2d(A_t, kernel_t, padding=1)  # 2D Laplacian convolution
        diffusion_B = d_B * F.conv2d(B_t, kernel_t, padding=1)

        reaction_A = A_t * B_t**2
        reaction_B = A_t * B_t**2

        feed_A = f * (1 - A_t)
        kill_B = (k + f) * B_t

        A_t += (diffusion_A - reaction_A + feed_A) * dt
        B_t += (diffusion_B + reaction_B - kill_B) * dt

        # Save interim results
        n = iters // 10
        if not i % n:
            interim_b.append(np.squeeze(B_t.numpy()).copy())

    A_out = np.squeeze(A_t.numpy())
    B_out = np.squeeze(B_t.numpy())

    return A_out, B_out, interim_b


if __name__=='__main__':
    main()
