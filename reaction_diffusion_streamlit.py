import time
import numpy as np
import pandas as pd
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
    random_noise = st.sidebar.checkbox('init_with_random_noise', value=False, key=session.run_id)
    if random_noise:
        rng_noise_amount_plh = st.sidebar.empty()
        rng_seed_plh = st.sidebar.empty()
    draw_square = st.sidebar.checkbox('init_with_square', value=True, key=session.run_id)
    if draw_square:
        square_ratio_plh = st.sidebar.empty()
    if (not random_noise) and (not draw_square):
        st.warning('Please initialize with random noise or square')
    iters = st.sidebar.slider('iterations', min_value=10, max_value=10000, step=10, value=6000, key=session.run_id)
    f = st.sidebar.slider('feed_rate', min_value=0.03, max_value=0.1, step=0.0001, value=0.055, key=session.run_id)  # 0.055
    k = st.sidebar.slider('kill_rate', min_value=0.03, max_value=0.1, step=0.0001, value=0.062, key=session.run_id)  # 0.062
    d_A = st.sidebar.slider('d_A', min_value=0.5, max_value=1.3, step=0.1, value=1.0, key=session.run_id)
    d_B = st.sidebar.slider('d_B', min_value=0.3, max_value=1.0, step=0.1, value=0.5, key=session.run_id)
    img_size = st.sidebar.slider('img_size', min_value=10, max_value=1000, step=1, value=200, key=session.run_id)
    cmap = st.sidebar.selectbox('plot_colormap', ['binary', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'], index=0,  key=session.run_id)
    dt = 1.0

    h = img_size
    w = img_size
    size = (h, w)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kernel = torch.tensor([
        [.05, .2, .05],
        [ .2, -1, .2 ],
        [.05, .2, .05]
    ], device=device)

    A = torch.ones(size, device=device)
    B = torch.zeros(size, device=device)

    # Add randomness to the pattern
    if random_noise:
        rng_seed = rng_seed_plh.slider('rng_seed', min_value=0, max_value=1000, step=10, value=40, key=session.run_id)
        rng_noise_amount = rng_noise_amount_plh.slider('rng_noise_amount', min_value=0.0, max_value=0.5, step=0.005, value=0.005, key=session.run_id)
        np.random.seed(rng_seed)

        B_ones_frac = rng_noise_amount  # fraction of B == 1
        B_ones_n = int(h + w * B_ones_frac)
        mask = (np.random.randint(0, h, B_ones_n), np.random.randint(0, w, B_ones_n))
        B[mask] = 1.0

    # Add a square in the center of the image
    if draw_square:
        square_ratio = square_ratio_plh.slider('square_ratio', min_value=0.05, max_value=1.0, step=0.05, value=0.1, key=session.run_id)
        h2 = h // 2
        w2 = w // 2
        _h = max(1, int(h * square_ratio))
        _w = max(1, int(w * square_ratio))
        B[h2-_h:h2+_h, w2-_w:w2+_w] = 1
        B[h2-_h + 1:h2+_h - 1, w2-_w + 1:w2+_w - 1] = 0  # fill the inner area with 0

    t0 = time.perf_counter()
    fig1 = plt.figure(figsize=(4,4))
    gs = fig1.add_gridspec(2, 5, wspace=0, hspace=0, top=0.5)

    # Simulate pattern
    with st.spinner('Processing...'):
        A, B, interim_b = simulate_torch(A, B, kernel, d_A, d_B, f, k, dt, iters)

    # Visualize interim results
    for i, b in enumerate(interim_b):
        plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(b, cmap=cmap)


    # Visualize the final result heatmap
    fig2 = plt.figure()
    plt.axis('off')
    plt.imshow(B, cmap=cmap)
    # plt.colorbar()
    st.pyplot(fig2, clear_figure=True)

    # Visualize interm results
    st.pyplot(fig1, clear_figure=True)

    st.write(f'Elapsed: {time.perf_counter() - t0:.2f} sec')
    st.write(f'Current parameters:', pd.DataFrame(dict(iters=iters, d_A=d_A, d_B=d_B, feed=f, kill=k, dt=dt), index=[0]))


def simulate_torch(A_t, B_t, kernel_t, d_A, d_B, f, k, dt, iters):

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
            interim_b.append(np.squeeze(B_t.cpu().numpy()).copy())

    A_out = np.squeeze(A_t.cpu().numpy())
    B_out = np.squeeze(B_t.cpu().numpy())

    return A_out, B_out, interim_b


if __name__=='__main__':
    main()
