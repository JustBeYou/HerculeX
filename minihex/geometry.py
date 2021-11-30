import numpy as np

def compute_hexagon(x, y, length):
    points = []
    for i in range(6):
        angle = i/3 * np.pi + np.pi/2
        points.append((x + length * np.cos(angle), y + length * np.sin(angle)))
    return points
    
def compute_hexagon_below_coefs(length):
    angle = -1/3 * np.pi
    dist = length * np.sqrt(3)
    return (dist * np.cos(angle), dist * np.sin(angle))
   
def translate_points(points, vx, vy):
    return [(x + vx, y + vy) for x, y in points]

def distance(xa, ya, xb, yb):
    return np.sqrt((xa-xb)**2 + (ya-yb)**2)