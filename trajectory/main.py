import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import ndimage

# Получаем путь к папке со скриптом
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "out", "h_*.npy")

files = sorted(glob.glob(path))

print(f"Ищу файлы: {path}")
print(f"Найдено файлов: {len(files)}")

traector = {}

for frame_num, file_path in enumerate(files):
    frame = np.load(file_path)
    labeled, num_objects = ndimage.label(frame > np.mean(frame) + np.std(frame))
    
    current_objects = []
    for obj_id in range(1, num_objects + 1):
        mask = labeled == obj_id
        if mask.sum() > 10:
            y, x = ndimage.center_of_mass(mask)
            current_objects.append((x, y))
    
    if frame_num == 0:
        for i, (x, y) in enumerate(current_objects):
            traector[i] = [(x, y)]
    else:
        used_objects = []
        
        for track_id, points in traector.items():
            last_x, last_y = points[-1]
            
            best_dist = 1000
            best_obj = None
            best_idx = -1
            
            for idx, (x, y) in enumerate(current_objects):
                if idx in used_objects:
                    continue
                dist = abs(x - last_x) + abs(y - last_y)
                if dist < best_dist:
                    best_dist = dist
                    best_obj = (x, y)
                    best_idx = idx
            
            if best_dist < 50 and best_obj is not None:
                traector[track_id].append(best_obj)
                used_objects.append(best_idx)
        
        for idx, (x, y) in enumerate(current_objects):
            if idx not in used_objects:
                if traector:
                    new_id = max(traector.keys()) + 1
                else:
                    new_id = 0
                traector[new_id] = [(x, y)]

print(f"Всего треков: {len(traector)}")

for id, points in traector.items():
    if len(points) > 1:
        coords = np.array(points)
        plt.plot(coords[:, 0], coords[:, 1], 'o-', color="blue")

plt.axis('equal')
plt.show()