import numpy as np
import socket
from skimage.measure import label
from scipy import ndimage

HOST = "84.237.21.36"
PORT = 5152
BUFFER_SIZE = 40002

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    client_socket.send(b"124ras1")
    client_socket.recv(10)
    
    for _ in range(10):
        client_socket.send(b'get')
        
        result = bytearray()
        while len(result) < BUFFER_SIZE:
            chunk = client_socket.recv(BUFFER_SIZE - len(result))
            if not chunk:
                break
            result.extend(chunk)
        
        height, width = result[0], result[1]
        pixel_data = np.frombuffer(result[2:BUFFER_SIZE], dtype="uint8")
        image = pixel_data.reshape(height, width)
        
        binary_mask = image > 0
        labeled_image = label(binary_mask)
        
        centers = []
        distance = 0.0
        for lbl in [1, 2]:
            if lbl not in labeled_image:
                distance = 0.0
                break
            region = labeled_image == lbl
            y, x = ndimage.center_of_mass(region)
            centers.append((x, y))
        else:
            x1, y1 = centers[0]
            x2, y2 = centers[1]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        
        client_socket.send(str(round(distance, 1)).encode())
        client_socket.recv(10)
        client_socket.send(b'beat')
        client_socket.recv(10)