import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random as rnd
import imageio
import numpy as np
import os
from own_functions import load_pictures, read_labels, test_parallel, metrics
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

NUMBER_OF_COLOURS = 3
NUMBER_OF_SHAPES = 10
SIDE_LENGTH = 28  # number of pixels on one side of picture


# for mixing sets
def shuffle_sets(x, y1, y2):
    new_x, new_y, new_y2 = [], [], []
    size_of_sets = len(x)
    list_with_idexes = list(range(size_of_sets))
    for value in x:
        index_to_get_index = rnd.randint(0, len(list_with_idexes) - 1)
        index = list_with_idexes.pop(index_to_get_index)
        new_x.append(x[index])
        new_y.append(y1[index])
        new_y2.append(y2[index])
    return np.array(new_x), np.array(tf.one_hot(new_y, NUMBER_OF_COLOURS)), np.array(
        tf.one_hot(new_y2, NUMBER_OF_SHAPES))


def save_label_to_file(label, name, root_dir):
    label_shape_filename = os.path.join(root_dir, "label_" + name + ".txt")
    os.makedirs(os.path.dirname(label_shape_filename), exist_ok=True)
    with open(label_shape_filename, "w") as file_shape:
        file_shape.write(str(label))


def save_labels_to_files(list_of_labels, root_dir):
    c = 0
    for label in list_of_labels:
        save_label_to_file(label, str(c), root_dir)
        c += 1


def get_random_colour():
    colour = [0, 0, 0]
    index = rnd.randint(0, 2)
    colour[index] = 255
    return colour, index


def get_random_colour_for_back(colour_foreground):
    colour = [0, 0, 0]
    index = rnd.randint(0, 1)
    if index == colour_foreground:
        random_one = 1
        if rnd.randint(0, 99) % 2 == 0:
            random_one = -1
        index = (index + random_one) % 3
    colour[index] = 255
    return colour


#  this function colors one picture with digit
def colour_digit(not_coloured_digit, colour_background=False, colour_background_randomly=False):
    colour_of_digit, colour_number = get_random_colour()
    if colour_background:
        one_back_colour = get_random_colour_for_back(colour_number)
    else:
        one_back_colour = [0, 0, 0]

    coloured_digit = []
    c = 0
    for row in not_coloured_digit:
        d = 0
        one_row = []
        for pixel in row:
            if pixel != 0:
                one_row.append(tf.convert_to_tensor(colour_of_digit, dtype=tf.uint8))
            elif colour_background_randomly:
                one_row.append(tf.convert_to_tensor(get_random_colour_for_back(colour_number), dtype=tf.uint8))
            else:
                one_row.append(tf.convert_to_tensor(one_back_colour, dtype=tf.uint8))
            d += 1
        one_row_tensor = tf.convert_to_tensor(one_row)
        coloured_digit.append(one_row_tensor)
        c += 1
    return coloured_digit, colour_number


#  this function colors the picture set with numbers, save it in root_dir, and save color labels
#  in colour_labels_di
def colour_dataset(dataset, root_dir, colour_background=False, colour_background_randomly=False,
                   colour_labels_dir=None):
    coloured_dataset = []
    colour_labels = []
    c = 0
    for img in dataset:
        coloured_img, colour_nr = colour_digit(img, colour_background, colour_background_randomly)
        coloured_dataset.append(coloured_img)
        colour_labels.append(colour_nr)
        imageio.imwrite(root_dir + "/example_" + str(c) + ".png", np.array(coloured_img))
        if colour_labels_dir is not None:
            save_label_to_file(colour_nr, str(c), colour_labels_dir)
        c += 1
    return coloured_dataset, colour_labels


def build_model_both():
    input_layer = Input(shape=(SIDE_LENGTH, SIDE_LENGTH, 3))
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255,
                                                                         input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3))(
        input_layer)
    layer_flatten = tf.keras.layers.Flatten()(normalization)
    layer_1 = Dense(SIDE_LENGTH * SIDE_LENGTH, activation="relu")(layer_flatten)

    layer_2 = Dense(SIDE_LENGTH, activation="relu")(layer_1)
    # layer_3 = Dense(SIDE_LENGTH, activation="relu")(layer_2)
    y1_output = Dense(NUMBER_OF_COLOURS, activation='softmax', name='colour_output')(layer_2)
    y2_output = Dense(NUMBER_OF_SHAPES, activation='softmax', name='shape_output')(layer_2)
    return Model(inputs=input_layer, outputs=[y1_output, y2_output])


def build_colour_model():
    input_layer = Input(shape=(SIDE_LENGTH, SIDE_LENGTH, 3))
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255,
                                                                         input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3))(
        input_layer)
    layer_flatten = tf.keras.layers.Flatten()(normalization)
    layer_1 = Dense(SIDE_LENGTH * SIDE_LENGTH, activation="relu")(layer_flatten)

    layer_2 = Dense(SIDE_LENGTH, activation="relu")(layer_1)
    # layer_3 = Dense(SIDE_LENGTH, activation="relu")(layer_2)
    y1_output = Dense(NUMBER_OF_COLOURS, activation='softmax', name='colour_output')(layer_2)
    return Model(inputs=input_layer, outputs=[y1_output])


def build_shape_model():
    input_layer = Input(shape=(SIDE_LENGTH, SIDE_LENGTH, 3))
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255,
                                                                         input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3))(
        input_layer)
    layer_flatten = tf.keras.layers.Flatten()(normalization)
    layer_1 = Dense(SIDE_LENGTH * SIDE_LENGTH, activation="relu")(layer_flatten)

    layer_2 = Dense(SIDE_LENGTH, activation="relu")(layer_1)
    # layer_3 = Dense(SIDE_LENGTH, activation="relu")(layer_2)
    y2_output = Dense(NUMBER_OF_SHAPES, activation='softmax', name='shape_output')(layer_2)
    return Model(inputs=input_layer, outputs=[y2_output])


# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images - 60000 black&white handwrite digits;
# test_images - 10000 black&white handwrite digits;
# colored below
# train_labels, test_labels -> shape labels

# =======================================================
# COLOR pictures and save them and save labels (of colors and shapes to files)

# colour_dataset(train_images, "digits_coloured_train", True, False, "digits_labels_colour_train")
# colour_dataset(test_images, "digits_coloured_test", True, False, "digits_labels_colour_test")

# colour_dataset(train_images, "colourful_background/digits_train", True, False, "colourful_background/labels_train")
# colour_dataset(test_images, "colourful_background/digits_test", True, False, "colourful_background/labels_test")

# colour_dataset(train_images, "colourful_background_random/digits_train", True, True,
# "colourful_background_random/labels_train")
# colour_dataset(test_images, "colourful_background_random/digits_test",
# True, True, "colourful_background_random/labels_test")
#
# save_labels_to_files(train_labels, "digits_labels_shapes_train")
# save_labels_to_files(test_labels, "digits_labels_shapes_test")
# =======================================================================

# -----------------------------------------------------------------------
# ----- this is for not change values in many places when we decide to explore a different type of pictures
type_picture = "c"
#  b - background is black
#  c - background is red or green or blue
#  r - background is red or green or blue (but pixels could be different)

# where are coloured pictures to train
pictures_train = {
    "b": "digits_coloured_train",
    "c": "colourful_background/digits_train",
    "r": "colourful_background_random/digits_train"
}

# where are coloured pictures to test
pictures_test = {
    "b": "digits_coloured_test",
    "c": "colourful_background/digits_test",
    "r": "colourful_background_random/digits_test"
}

# where are labels of colors to train
labels_train = {
    "b": "digits_labels_colour_train",
    "c": "colourful_background/labels_train",
    "r": "colourful_background_random/labels_train"
}

# where are labels of colors to test
labels_test = {
    "b": "digits_labels_colour_test",
    "c": "colourful_background/labels_test",
    "r": "colourful_background_random/labels_test"
}

coloured_train_images = load_pictures(pictures_train[type_picture])  # train images load to array of array
coloured_test_images = load_pictures(pictures_test[type_picture])  # test images load to array of array
#
labels_colour_train, labels_shapes_train, labels_both_train = read_labels(labels_train[type_picture],
                                                                          "digits_labels_shapes_train")
# "digits_labels_shapes_train" is hardcoded because labels for shapes are the same for all types of pictures

labels_colour_test, labels_shapes_test, labels_both_test = read_labels(labels_test[type_picture],
                                                                       "digits_labels_shapes_test")
# "digits_labels_shapes_train" is hardcoded because labels for shapes are the same for all types of pictures

# make sets of X (expected inputs to neural nets)
x_train = np.array(coloured_train_images)
x_test = np.array(coloured_test_images)

# make sets of Y (expected responses from neural nets)
y_set_colour_train = np.array(tf.one_hot(labels_colour_train, NUMBER_OF_COLOURS))
y_set_shape_train = np.array(tf.one_hot(labels_shapes_train, NUMBER_OF_SHAPES))
y_set_both_train = [y_set_colour_train, y_set_shape_train]

y_set_colour_test = np.array(tf.one_hot(labels_colour_test, NUMBER_OF_COLOURS))
y_set_shape_test = np.array(tf.one_hot(labels_shapes_test, NUMBER_OF_SHAPES))
y_set_both_test = [y_set_colour_test, y_set_shape_test]


def train_and_test():
    # ======================================
    # MODELS
    # --------- a model for recognizing color and shape (which will say, for example, "0, 7" - red seven)
    model_both = build_model_both()
    # --------- a color recognition model (which will say, for example, 0 -red)
    model_colour = build_colour_model()
    # --------- a model for recognizing a shape (which will say, for example, 7 - seven)
    model_shape = build_shape_model()
    # ======================================
    # --- compiling
    model_both.compile(optimizer='adam',
                       loss=
                       {
                           'colour_output': tf.keras.losses.CategoricalCrossentropy(),
                           'shape_output': tf.keras.losses.CategoricalCrossentropy()
                       }
                       ,
                       metrics=['accuracy', tf.metrics.Precision()])
    model_colour.compile(optimizer='adam',
                         loss=
                         {
                             'colour_output': tf.keras.losses.CategoricalCrossentropy(),
                         },
                         metrics=['accuracy', tf.metrics.Precision()])
    model_shape.compile(optimizer='adam',
                        loss=
                        {
                            'shape_output': tf.keras.losses.CategoricalCrossentropy()
                        },
                        metrics=['accuracy', tf.metrics.Precision()])

    #  mix sets
    x_train_sh, y_set_colour_train_sh, y_set_shape_train_sh = shuffle_sets(coloured_train_images, labels_colour_train,
                                                                           labels_shapes_train)
    print("Without training one network (both features): ==================")
    print(metrics(model_both.predict(x_test), y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES))
    result = model_both.evaluate(x_test, y_set_both_test)
    print(dict(zip(model_both.metrics_names, result)))

    #  train one network (both features)
    history = model_both.fit(x_train_sh, [y_set_colour_train_sh, y_set_shape_train_sh], epochs=3)

    #  look at how shape_output_accuracy is changed in this model
    plt.interactive(False)
    plt.plot(history.history['shape_output_accuracy'])
    plt.show()

    #  mix sets again
    x_train_sh, y_set_colour_train_sh, y_set_shape_train_sh = shuffle_sets(coloured_train_images, labels_colour_train,
                                                                           labels_shapes_train)
    print("Without training parallel (accuracy):-------------------")
    print(test_parallel(
        model_colour, model_shape, x_test, y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES))  # accuracy
    result = model_shape.evaluate(x_test, y_set_shape_test)
    print("Without training (shape model):-------------------")
    print(dict(zip(model_shape.metrics_names, result)))

    #  train model colour
    model_colour.fit(x_train_sh, y_set_colour_train_sh, epochs=1)

    #  train model shape
    history = model_shape.fit(x_train_sh, y_set_shape_train_sh, epochs=3)

    #  look at how accuracy is changed in this model
    plt.interactive(False)
    plt.plot(history.history['accuracy'])
    plt.show()

    return metrics(model_both.predict(x_test), y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES), test_parallel(
        model_colour, model_shape, x_test, y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES)


def test_and_train_times(iterations=5):
    boths = []
    pars = []
    for i in range(iterations):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(i)
        b, p = train_and_test()
        boths.append(b)
        pars.append(p)
    print(type_picture)
    print("------------ one (both in one)")
    print(sum(boths) / iterations)
    print(boths)
    print("------------ pararel")
    print(sum(pars) / iterations)
    print(pars)


both, par = train_and_test()
print("Accuracy one network (both features): " + str(both))
print("Accuracy parallel network: " + str(par))

# test_and_train_times()
