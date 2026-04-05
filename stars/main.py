import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os

base_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_path, 'stars.npy')

image = np.load(path)

labeled, num = label(image, np.ones((3,3)))

cnt = 0

for i in range(1, num + 1):
    coords = np.argwhere(labeled == i)

    y_min, x_min = coords.min(axis = 0)
    y_max, x_max = coords.max(axis = 0)

    area = (y_max - y_min) * (x_max - x_min)
    obj_area = len(coords)
    
    if obj_area / area < 0.75:
        cnt += 1

print(f"Всего звёзд: {cnt}")

plt.imshow(labeled, cmap = "gray")
plt.show()