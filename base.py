from time import perf_counter
from PIL import Image
import numpy as np

# Image settings
ASPECT_RATIO = 16/9
WIDTH = 600
HEIGHT = int(WIDTH / ASPECT_RATIO)
pixels = WIDTH * HEIGHT
#Camera settings
viewport_height = 2
viewport_width = ASPECT_RATIO * viewport_height
focal_length = 1

origin = np.array([0,0,1])
hittable_spheres = [
    {'center': np.array([0,0,-1]), 'radius': 0.5},
    {'center': np.array([0,-100.5, -1]), 'radius': 100}
]

def hit_sphere(center, radius, origins:np.ndarray, directions:np.ndarray) -> np.ndarray:
    oc = origins - center
    a = np.sum(directions * directions, axis=1)
    half_b = np.sum(oc * directions, axis=1)
    c = np.sum(oc * oc, axis=1) - radius*radius
    discriminant = half_b*half_b - a*c
    t1 = (-half_b - np.sqrt(discriminant, where=discriminant>=0))/a
    t2 = (-half_b + np.sqrt(discriminant, where=discriminant>=0))/a
    t = np.where(t1 < 0, t2, t1)
    t = np.where(t < 0, 0, t)
    return (discriminant >= 0) * t

def normalize(a : np.ndarray) -> np.ndarray:
    return ((1/np.linalg.norm(a, axis=1)) * a.T).T
def ray_color(origins : np.ndarray, directions : np.ndarray) -> np.ndarray:
    unit_direction = normalize(directions)
    t_array = np.multiply(np.add(unit_direction[:, 1],1),0.5)
    colors = ((1-t_array)*np.tile([1.0,1.0,1.0], pixels).reshape(-1, 3).T + 
                (t_array * np.tile([0.5,0.7,1.0], pixels).reshape(-1, 3).T)).T

    t_min = np.zeros(pixels)
    centers_min = np.zeros((pixels, 3))
    radius_min = np.ones(pixels)
    for sphere in hittable_spheres:
        t = hit_sphere(sphere['center'], sphere['radius'], origins, directions)
        t_min = np.where(t_min <= 0, t, t_min)
        t_min = np.minimum(t_min, t) + (1-(t*t_min > 0))*t_min
        new_min = (t == t_min) * (t_min > 0)
        vector_min = np.repeat(new_min, 3).reshape(-1, 3)
        centers_min = (1-vector_min)*centers_min + (vector_min * sphere['center'])
        radius_min = (1-new_min)*radius_min + (new_min * sphere['radius'])

    hits = np.repeat(t_min>0, 3).reshape(-1, 3)
    r_at_t = origins + np.repeat(t_min,3).reshape(-1, 3) * directions
    N = (r_at_t - centers_min) / np.repeat(radius_min, 3).reshape(-1, 3)
    N = 0.5*(N+1)
    colors = (1-hits)*colors + hits*N
    return colors
def rgb_color(a : np.ndarray) -> np.ndarray:
    '''
     Creating a numpy array from an array of range [0,1] to an array of [0,255]
    '''
    return (np.array(a) * 255.99).astype(np.uint8)
image_array = []
t1 = perf_counter()

x = np.linspace(-viewport_width/2, viewport_width/2, WIDTH)
# y = np.linspace(-viewport_height/2, viewport_height/2, HEIGHT)
y = np.linspace(viewport_height/2, -viewport_height/2, HEIGHT) # Modified to be consistent with the original.
screen = np.stack([np.tile(x, HEIGHT), np.repeat(y, WIDTH), np.zeros(pixels)]).T

# Ray Tracing
origin_array = np.zeros((pixels, 3))
direction_array = screen - origin
colors = ray_color(origin_array, direction_array)

image_array = colors.reshape(HEIGHT, WIDTH, 3)
image_array = rgb_color(image_array)

im = Image.fromarray(image_array)
im.save('image.jpg')        

t2 = perf_counter()
print(f"Execution Time: {t2-t1}")