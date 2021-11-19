import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v1 import SGD


def rgb2int(rgb): return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]


def test_parallel(model_for_colour, model_for_shape, x_test, y_test):
    y_colour_predict = model_for_colour.predict(x_test) # predict colour
    y_shape_predict = model_for_shape.predict(x_test)  # predict shape
    y_predict = []  # here will be cartesian product of result from colour and shape

    for i in range(len(y_colour_predict)):
        y_predict.append([y_colour_predict[i], y_shape_predict[i]])  # cartesian product
    y_predict_np = np.squeeze(np.array(y_predict))

    for i in range(len(y_test)):
        # print(y_predict_np[i])  # TODO handmade metrics
        # print(y_test[i])
        # print(y_test[i] == y_predict_np[i])
        print(y_colour_predict[i])
    print(y_predict_np.shape)
    print(y_test.shape)

    # =========================================================== READ DATA


# read colour labels from files to array >>labels_from_color<<
root_dir = "labels_colours"
labels_from_color = []
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        with open(root_dir + "/" + file, "r") as label_color:
            labels_from_color.append(int(label_color.readline()))

# read shape labels from files to array >>labels_from_shapes<<
# make array with shape&&colour >>labels_from_both<<
root_dir = "labels_shapes"
labels_from_shapes = []
labels_from_both = []
i = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        with open(root_dir + "/" + file, "r") as label_shape:
            labels_from_shapes.append(int(label_shape.readline()))
            print(labels_from_color[i])
            print(
                type(labels_from_color[i])
            )
            print(labels_from_shapes[i])
            print(
                type(labels_from_shapes[i])
            )
            labels_from_both.append([labels_from_color[i], labels_from_shapes[i]])
            i += 1

root_dir = "one_folder"

# make set of X (inputs to neural nets)
x_set_array = []
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        img = Image.open(root_dir + "/" + file).convert("RGB")
        pixels = img.load()
        example_two_dim = []
        img_width, img_height = img.size  # 3 x 3
        for i in range(img_height):
            example = []
            for j in range(img_width):
                example.append(rgb2int(pixels[i, j]))
            example_two_dim.append(example)
        x_set_array.append(example_two_dim)

x_train_set = np.array(x_set_array)
x_test_set = np.array(x_set_array)  # TODO data should be different than in x_train_set

# make sets of Y (expected responses from neural nets)
y_train_set_colour = np.array(labels_from_color)
y_train_set_shape = np.array(labels_from_shapes)
y_train_set_both = np.array(labels_from_both)
y_test_set_both = np.array(labels_from_both)  # TODO data should be different than in y_train_set
# ===========================================================

# ======================================
# MODELS, so far they are so simple, small
# --------- a model for recognizing color and shape (which will say, for example, "yes, that's a red square")
model_both = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(3, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

# --------- a color recognition model (which will say, for example, 0 -red)
model_colour = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(3, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

# --------- a model for recognizing a shape (which will say, for example, 1 - circle)
model_shape = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(3, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
])
# ======================================

# --- training
model_both.compile(optimizer='adam',
                   loss=tf.keras.losses.mean_squared_logarithmic_error,
                   metrics=['accuracy'])

model_both.fit(x_train_set, y_train_set_both, epochs=10)

model_colour.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])

model_colour.fit(x_train_set, y_train_set_colour, epochs=10)

model_shape.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

model_shape.fit(x_train_set, y_train_set_shape, epochs=10)

model_both.evaluate(x_test_set, y_test_set_both)  # auto. eval. of the results of the network trained for both features

test_parallel(model_colour, model_shape, x_test_set, y_test_set_both)  # manual testing network consisting of two nets
print(y_train_set_colour[5])
print(model_both.predict(x_test_set)[5])
