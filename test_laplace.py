import numpy as np
import scipy.signal as sl
import matplotlib.pyplot as plt

size = (10, 10)
U = np.random.normal(size=size)
print(U)

# Approach from git implementation
dx = 1
U1 = (U[0:-2,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] + U[2:,1:-1] - 4*U[1:-1,1:-1]) / dx**2


# My approach based on 2d convolution
kernel = np.asarray([
    [.05, .2, .05],
    [ .2, -1, .2 ],
    [.05, .2, .05]
])  # 3x3 convolution

U2 = sl.convolve2d(U, kernel, mode='same', boundary='wrap')
# align with U1
U2 = U2[1:-1, 1:-1]

print(U1)
print(U2)

plt.subplot(121)
plt.imshow(U1)
plt.subplot(122)
plt.imshow(U2)
plt.show()
