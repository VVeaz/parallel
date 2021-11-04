import os
from PIL import Image
import numpy as np


def rgb2int(rgb): return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]


# read colour labels from files to array >>labels_from_color<<
root_dir = "labels_colours"
labels_from_color = []
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        with open(root_dir + "/" + file, "r") as label_color:
            labels_from_color.append(int(label_color.read(1)))

# read shape labels from files to array >>labels_from_shapes<<
# make array with shape&&colour >>labels_from_both<<
root_dir = "labels_shapes"
labels_from_shapes = []
labels_from_both = []
i = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        with open(root_dir + "/" + file, "r") as label_shape:
            labels_from_shapes.append(int(label_shape.read(1)))
            labels_from_both.append(labels_from_color[i] * labels_from_shapes[i])
            i += 1


root_dir = "one_folder"

# make set of X (inputs to neural nets)
x_set_array = []
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        img = Image.open(root_dir + "/" + file).convert("RGB")
        pixels = img.load()
        example = []
        img_width, img_height = img.size  # 3 x 3
        for i in range(img_height):
            for j in range(img_width):
                example.append(rgb2int(pixels[i, j]))
        x_set_array.append(example)

x_set = np.array(x_set_array)

# make sets of Y (expected responses from neural nets)
y_set_colour = np.array(labels_from_color)
y_set_shape = np.array(labels_from_shapes)
y_set_both = np.array(labels_from_both)

print(x_set.shape)

print(y_set_colour.shape)
print(y_set_shape.shape)
print(y_set_both.shape)
