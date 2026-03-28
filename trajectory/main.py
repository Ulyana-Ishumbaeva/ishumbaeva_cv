import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import ndimage

path = "out/h_*.npy"

files = sorted(glob.glob(path))

traector = {}

for num, file_path in enumerate(files):
    frame = np.load(file_path)
    labeled, num_obj = ndimage.label(frame > np.mean(frame) + np.std(frame))
    
    crnt_obj = []
    for obj_id in range(1, num_obj + 1):
        mask = labeled == obj_id
        if mask.sum() > 10:
            y, x = ndimage.center_of_mass(mask)
            crnt_obj.append((x, y))
    
    if num == 0:
        for i, (x, y) in enumerate(crnt_obj):
            traector[i] = [(x, y)]
    else:
        used_obj = []
        
        for id, points in traector.items():
            last_x, last_y = points[-1]
            
            best_dist = 1000
            best_obj = None
            best_idx = -1
            
            for idx, (x, y) in enumerate(crnt_obj):
                if idx in used_obj:
                    continue
                dist = abs(x - last_x) + abs(y - last_y)
                if dist < best_dist:
                    best_dist = dist
                    best_obj = (x, y)
                    best_idx = idx
            
            if best_dist < 50 and best_obj is not None:
                traector[id].append(best_obj)
                used_obj.append(best_idx)
        
        for idx, (x, y) in enumerate(crnt_obj):
            if idx not in used_obj:
                if traector:
                    new_id = max(traector.keys()) + 1
                else:
                    new_id = 0
                traector[new_id] = [(x, y)]

for id, points in traector.items():
    if len(points) > 1:
        coords = np.array(points)
        plt.plot(coords[:, 0], coords[:, 1], 'o-')

plt.axis('equal')
plt.show()