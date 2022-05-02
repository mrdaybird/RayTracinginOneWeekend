from time import perf_counter
import os
from PIL import Image
import numpy as np

# Image settings
ASPECT_RATIO = 16/9
WIDTH = 400
HEIGHT = int(WIDTH / ASPECT_RATIO)

#Camera settings
viewport_height = 2
viewport_width = ASPECT_RATIO * viewport_height
focal_length = 1

origin = np.array([0,0,-1])
def hit_sphere(center, radius, origins:np.ndarray, directions:np.ndarray) -> np.ndarray:
    oc = origins - center
    a = np.sum(directions * directions, axis=1)
    b = 2 * np.sum(oc * directions, axis=1)
    c = np.sum(oc * oc, axis=1) - radius*radius
    discriminant = b*b - 4*a*c
    return discriminant > 0

def normalize(a : np.ndarray) -> np.ndarray:
    return ((1/np.linalg.norm(a, axis=1)) * a.T).T
def ray_color(origins : np.ndarray, directions : np.ndarray) -> np.ndarray:
    unit_direction = normalize(directions)
    t_array = np.multiply(np.add(unit_direction[:, 1],1),0.5)
    colors = ((1-t_array)*np.tile([1.0,1.0,1.0], WIDTH * HEIGHT).reshape(-1, 3).T + 
                (t_array * np.tile([0.5,0.7,1.0], WIDTH*HEIGHT).reshape(-1, 3).T)).T

    hits = hit_sphere(np.array([0,0,-1]), 0.5, origins, directions)
    hits = np.repeat(hits, 3).reshape(-1, 3)
    colors = (1-hits)*colors + hits*np.tile([1,0,0], 90000).reshape(90000, 3)
    return colors
def rgb_color(a : np.ndarray) -> np.ndarray:
    '''
     Creating a numpy array from an array of range [0,1] to an array of [0,255]
    '''
    return (np.array(a) * 255.99).astype(np.uint8)
image_array = []
t1 = perf_counter()

x = np.linspace(-viewport_width/2, viewport_width/2, WIDTH)
y = np.linspace(-viewport_height/2, viewport_height/2, HEIGHT)
screen = np.stack([np.tile(x, HEIGHT), np.repeat(y, WIDTH), np.zeros(WIDTH * HEIGHT)]).T

origin_array = np.zeros((90000, 3))
direction_array = screen - origin
colors = ray_color(origin_array, direction_array)

image_array = colors.reshape(HEIGHT, WIDTH, 3)

image_array = rgb_color(image_array)

im = Image.fromarray(image_array)
im.save('image.jpg')        

t2 = perf_counter()
print(f"Execution Time: {t2-t1}")