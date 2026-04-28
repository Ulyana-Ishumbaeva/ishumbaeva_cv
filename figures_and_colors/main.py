import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.color import rgb2hsv
from pathlib import Path

script_dir = Path(__file__).parent
image_path = script_dir / "balls_and_rects.png"
image = imread(str(image_path))

hsv = rgb2hsv(image)
v = hsv[:, :, 2]
h = hsv[:, :, 0]

labeled = label(v > 0.3)
print(f"Количество всех фигур на изображении: {labeled.max()}")

figures = {"circle": {}, "rectangle": {}}

for region in regionprops(labeled):
    mask = labeled == region.label
    mean_hue = h[mask].mean()

    circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)

    extent = region.extent

    if extent < 0.9 and circularity > 0.85:
        shape_type = "circle"
    else:
        shape_type = "rectangle"

    figures[shape_type][mean_hue] = figures[shape_type].get(mean_hue, 0) + 1

for shape, colors in figures.items():
    print(f"{shape}:")
    for hue, count in colors.items():
        print(f"  оттенок {hue:.3f}: {count} шт.")
    print()

plt.imshow(h, cmap="gray")
plt.title("Карта оттенков (H)")
plt.axis("off")
plt.show()