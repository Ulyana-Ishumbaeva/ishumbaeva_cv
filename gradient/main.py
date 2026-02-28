import numpy as np
import matplotlib.pyplot as plt

size = 100
image = np.zeros((size, size, 3), dtype="uint8")


color1 = [0, 128, 255]
color2 = [255, 128, 0]

def lerp(a, b, t):
    return a + (b - a) * t

for i in range(size):
    for j in range(size):
        t = (i/(size-1) + j/(size-1)) / 2

        r = lerp(color1[0], color2[0], t)
        g = lerp(color1[1], color2[1], t)
        b = lerp(color1[2], color2[2], t)

        image[i, j] = [int(r), int(g), int(b)]

plt.imshow(image)
plt.show()