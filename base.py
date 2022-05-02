from time import perf_counter
import os

from pyrr import Vector3
from PIL import Image
import numpy as np

cwd = os.getcwd()
log_file = open(cwd + '\\' + 'logfile.txt', 'w')

# Image settings
ASPECT_RATIO = 16/9
WIDTH = 400
HEIGHT = int(WIDTH / ASPECT_RATIO)

#Camera settings
viewport_height = 2
viewport_width = ASPECT_RATIO * viewport_height
focal_length = 1

origin = Vector3([0,0,-1])
horizontal = Vector3([viewport_width, 0, 0])
vertical = Vector3([0,viewport_height,0])
lower_left_corner = origin - horizontal / 2 - vertical/2 - Vector3([0,0,focal_length])  

class ray:
    def __init__(self, origin:Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction

    def __matmul__(self, t):
        return self.origin + t * self.direction
    
def ray_color(r: ray):
    unit_direction = r.direction.normalized
    #[-1,1] --> [0,2] --> [0,1]
    t = 0.5*(unit_direction.y + 1.0)
    return (1-t) * Vector3([1,1,1]) + t*Vector3([0.5, 0.7, 1.0])
def color_rgb(r,g,b):
    '''
        Converted color in [0,1] to [0,255] 
    '''
    return [r,g,b]*255.99

image_array = []
t1 = perf_counter()
for i in np.linspace(-viewport_height/2, viewport_height/2, HEIGHT):
    row = []
    _remaining_lines = HEIGHT - int((i/viewport_height + 1/2)*(HEIGHT)) 
    print(f'Lines remaining: {_remaining_lines}', file=log_file, flush=True)
    for j in np.linspace(-viewport_width/2, viewport_width/2, WIDTH):
        # r = j/(WIDTH -1)
        # g = i/(HEIGHT -1)
        # b = 0.25
        # row.append([r,g,b])

        point = Vector3([j, i, 0])

        r = ray(origin, point - origin)
        pixel_color = ray_color(r)
        row.append(pixel_color)
    image_array.append(row)

# Creating a numpy array from [0,1] colored to [0,255]
image_array = (np.array(image_array) * 255.99).astype(np.uint8)
im = Image.fromarray(image_array)
im.save('image.jpg')        

t2 = perf_counter()
print('Done.', file=log_file)
print(f"Execution Time: {t2-t1}")