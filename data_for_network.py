import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense

DATA_SET_SIZE = 3072
SPLIT = int(DATA_SET_SIZE * 0.8)
NUMBER_OF_COLOURS = 3
NUMBER_OF_SHAPES = 4


def read_labels(dir_colour, dir_shape):
    # read colour labels from files to array >>labels_from_color<<
    root_dir = dir_colour
    labels_col = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            with open(root_dir + "/" + file, "r") as label_color:
                labels_col.append(int(label_color.readline()))

    # read shape labels from files to array >>labels_from_shapes<<
    # make array with shape&&colour >>labels_from_both<<
    root_dir = dir_shape
    labels_shap = []
    labels_both = []
    i = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            with open(root_dir + "/" + file, "r") as label_shape:
                labels_shap.append(int(label_shape.readline()))
                labels_both.append([labels_col[i], labels_shap[i]])
                i += 1
    return labels_col, labels_shap, labels_both


def load_pictures(root_dir):
    x = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            img = Image.open(root_dir + "/" + file).convert("RGB")
            pixels = img.load()
            example_two_dim = []
            img_width, img_height = img.size  # 3 x 3
            for i in range(img_height):
                example = []
                for j in range(img_width):
                    # example.append(rgb2int(pixels[i, j]))
                    example.append(pixels[i, j])
                example_two_dim.append(example)
            x.append(example_two_dim)
    return x


def build_model_both():
    input_layer = Input(shape=(3, 3, 3))
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(3, 3, 3))(input_layer)
    layer_flatten = tf.keras.layers.Flatten()(normalization)
    layer_1 = Dense(128, activation="relu")(layer_flatten)
    # layer_2 = Dense(128, activation="relu")(layer_1)
    y1_output = Dense(NUMBER_OF_COLOURS, activation='softmax', name='colour_output')(layer_1)
    y2_output = Dense(NUMBER_OF_SHAPES, activation='softmax', name='shape_output')(layer_1)
    return Model(inputs=input_layer, outputs=[y1_output, y2_output])


def build_colour_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(3, 3, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(3, 3, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUMBER_OF_COLOURS, activation='softmax')
    ])


def build_shape_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(3, 3, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(3, 3, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUMBER_OF_SHAPES, activation='softmax')
    ])


def test_parallel(model_for_colour, model_for_shape, x_test, y_test):
    y_colour_predict = tf.one_hot(np.argmax(model_for_colour.predict(x_test), axis=-1),
                                  NUMBER_OF_COLOURS)  # predict colour
    y_shape_predict = tf.one_hot(np.argmax(model_for_shape.predict(x_test), axis=-1), NUMBER_OF_SHAPES)  # predict shape
    # y_predict = [y_colour_predict, y_shape_predict]  # here will be cartesian product of result from colour and shape

    for i in range(len(y_test[0])):
        print(y_colour_predict[i])  # TODO handmade metrics
        print(y_shape_predict[i])
        print(y_test[0][i])
        print(y_test[1][i])
        # print(y_test[0][i] == y_colour_predict[i])
        # print(y_test[i] == y_predict_np[i])
        print("-------")


# =========================================================== READ DATA
labels_from_color, labels_from_shapes, labels_from_both = read_labels("labels_colours", "labels_shapes")

# make set of X (inputs to neural nets)
x_set_array = load_pictures("one_folder")

# split them to training and test
x_train_set = np.array(x_set_array[:SPLIT])
x_test_set = np.array(x_set_array[SPLIT:]) 

# make sets of Y (expected responses from neural nets)
y_set_colour = np.array(tf.one_hot(labels_from_color, NUMBER_OF_COLOURS))
y_set_shape = np.array(tf.one_hot(labels_from_shapes, NUMBER_OF_SHAPES))

# split them to train and test
y_train_set_colour = y_set_colour[:SPLIT]
y_train_set_shape = y_set_shape[:SPLIT]
y_train_set_both = [y_train_set_colour, y_train_set_shape]
y_test_set_both = [y_set_colour[SPLIT:], y_set_shape[SPLIT:]]
# ===========================================================

# ======================================
# MODELS
# --------- a model for recognizing color and shape (which will say, for example, "0 1" - red circle)
model_both = build_model_both()
# --------- a color recognition model (which will say, for example, 0 -red)
model_colour = build_colour_model()
# --------- a model for recognizing a shape (which will say, for example, 1 - circle)
model_shape = build_shape_model()
# ======================================
# --- compiling
model_both.compile(optimizer='adam',
                   loss={'colour_output': tf.keras.losses.CategoricalCrossentropy(),
                         'shape_output': tf.keras.losses.CategoricalCrossentropy()
                         },
                   metrics=['accuracy', tf.metrics.Precision()])
model_colour.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(),
                     metrics=['accuracy'])
model_shape.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
# --- training
model_both.fit(x_train_set, y_train_set_both, epochs=10)
model_colour.fit(x_train_set, y_train_set_colour, epochs=10)
model_shape.fit(x_train_set, y_train_set_shape, epochs=10)

result = model_both.evaluate(x_test_set, y_test_set_both)
# auto. eval. of the results of the network trained for both features
print(dict(zip(model_both.metrics_names, result)))

test_parallel(model_colour, model_shape, x_test_set, y_test_set_both)  # manual testing network consisting of two nets
