import os
import matplotlib.pyplot as plt

from scipy.io import loadmat

FILENAME = 'data_25'

mat = loadmat(os.path.join('data', FILENAME))

# Plot a number of samples from the data.
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(mat[FILENAME][..., i], cmap='gray')

plt.show()
