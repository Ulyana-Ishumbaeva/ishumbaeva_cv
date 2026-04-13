import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0]+2, shape[1]+2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) -1

def count_lines(region):
    shape = region.image.shape
    image = region.image
    vlines = (np.sum(image, 0) / shape[0] == 1).sum()
    hlibes = (np.sum(image, 1) / shape[1]==1).sum()
    return vlines, hlibes

def symmetry(region, transpose = False):
    image = region.image
    if transpose:
        image = image.T
    shape = image.shape
    top = image[:shape[0]//2]
    if shape[0] % 2 != 0:
        bottom = image[shape[0]//2+1:]
    else:
        bottom = image[-shape[0]//2:]
    bottom = bottom[::-1]
    result = bottom == top
    return result.sum() / result.size

def extractor(region):
    cy,cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    holes = count_holes(region)
    v,h= count_lines(region)
    v /= region.image.shape[1]
    h /= region.image.shape[0]
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] / region.image.shape[1]
    v_sym = symmetry(region)
    h_sym = symmetry(region, transpose=True)
    return np.array([region.area/region.image.size, cy,cx,perimeter, holes, v, h, eccentricity, aspect, v_sym, h_sym])

def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10**6
    for symbol, t in templates.items():
        d = ((t-features)**2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result

template = imread("alphabet-small.png")[:,:,:-1]
print(template.shape)
template = template.sum(2)
binary = template != 765

labeled = label(binary)
props = regionprops(labeled)

templates = {}
for region,symbol in zip(props, ["8", "O", "A", "B", "1", "W", "X", "*", "/", "-"]):
    templates[symbol] = extractor(region)
print(templates)

print(classificator(props[5], templates))

print(type(props[0]))
print(props[0].area, props[0].centroid, props[0].label)

image = imread("alphabet.png")[:,:,:-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print(alabeled.max())

aprops = regionprops(alabeled)
result = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region, templates)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path/f"image_{region.label}.png")
print(result)

plt.imshow(alabeled)
plt.show()
#{'/': 21, 'B': 25, '-': 20, '8': 23, 'A': 21, '1': 31, 'W': 12, '*': 22, 'O': 10, 'X': 15}