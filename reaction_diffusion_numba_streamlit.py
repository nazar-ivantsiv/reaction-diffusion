import time
import numpy as np
import pandas as pd
import numba as nb

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
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'], index=2,  key=session.run_id)
    dt = 1.0

    h = img_size
    w = img_size
    size = (h, w)
    kernel = np.asarray([
    [.05, .2, .05],
    [ .2, -1, .2 ],
    [.05, .2, .05]
    ])

    np.random.seed(42)
    A = np.ones(size) + np.random.rand(*size) * 0.1
    B = np.zeros(size) + np.random.rand(*size) * 0.1

    def generate_numba_function():

        @nb.njit(fastmath=True, nogil=True, parallel=True)
        def function(a, b):
            diffusion_a = np.zeros_like(a)
            diffusion_b = np.zeros_like(b)
            for i in nb.prange(y):
                for j in nb.prange(x):
                    for mi in nb.prange(m):
                        for mj in nb.prange(m):
                            diffusion_a[i, j] += a[i + mi - m // 2, j + mj - m // 2] * kernel[mi, mj]
                            diffusion_b[i, j] += b[i + mi - m // 2, j + mj - m // 2] * kernel[mi, mj]
                    diffusion_a[i, j] = diffusion_a[i, j] * d_A
                    diffusion_b[i, j] = diffusion_b[i, j] * d_B
            for i in nb.prange(y):
                for j in nb.prange(x):
                    reaction_a = a[i][j] * b[i][j] ** 2
                    reaction_b = a[i][j] * b[i][j] ** 2
                    a[i][j] += (diffusion_a[i, j] - reaction_a + f * (1 - a[i][j])) * dt
                    b[i][j] += (diffusion_b[i, j] + reaction_b - (k + f) * b[i][j]) * dt
            return a, b

        m, _ = kernel.shape
        x, y = A.shape
        x = x - m + 1
        y = y - m + 1
        function(np.zeros_like(A), np.zeros_like(A))
        return function

    simulate_numba = generate_numba_function()

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

        for i in range(iters):
            A, B = simulate_numba(A, B)
            # Visualize interm results
            n = iters // 10
            if not i % n:
                plt.subplot(2, 5, i // n + 1)
                plt.axis('off')
                plt.imshow(B, cmap=cmap)

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


if __name__=='__main__':
    main()
