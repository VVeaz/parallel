import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random as rnd
import imageio
import numpy as np
import os
import cv2
import time
from PIL import Image
from tensorflow.keras.utils import to_categorical
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


def to_rgb(not_coloured_digit):
    one_back_colour = [0, 0, 0]
    one_forground = [255, 255, 255]
    coloured_digit = []
    c = 0
    for row in not_coloured_digit:
        d = 0
        one_row = []
        for pixel in row:
            if pixel != 0:
                one_row.append(tf.convert_to_tensor(one_forground, dtype=tf.uint8))
            else:
                one_row.append(tf.convert_to_tensor(one_back_colour, dtype=tf.uint8))
            d += 1
        one_row_tensor = tf.convert_to_tensor(one_row)
        coloured_digit.append(one_row_tensor)
        c += 1
    return coloured_digit


def save_gray_set(dataset, root_dir):
    coloured_dataset = []
    colour_labels = []
    c = 0
    for img in dataset:
        coloured_img = to_rgb(img)
        coloured_dataset.append(coloured_img)
        imageio.imwrite(root_dir + "/example_" + str(c) + ".png", np.array(coloured_img))
        c += 1


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


def gray_to_rgb(picture_set):
    rgb_set = []
    for pic in picture_set:
        rgb_set.append(cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB))
    return rgb_set


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
# save_gray_set(train_images, "black_white_digits")
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
type_picture = "gb"
#  b - background is black
#  c - background is red or green or blue
#  r - background is red or green or blue (but pixels could be different)

# where are coloured pictures to train
pictures_train = {
    "b": "digits_coloured_train",
    "c": "colourful_background/digits_train",
    "r": "colourful_background_random/digits_train",
    "gb": "generated_colorful_black_bg/digits_train",
    "gc": "generated_colorful_color_bg/digits_train",
    "gr": "generated_colorful_color_bg_random/digits_train"
}

# where are coloured pictures to test
pictures_test = {
    "b": "digits_coloured_test",
    "c": "colourful_background/digits_test",
    "r": "colourful_background_random/digits_test",
    "gb": "generated_colorful_black_bg/digits_test",
    "gc": "generated_colorful_color_bg/digits_test",
    "gr": "generated_colorful_color_bg_random/digits_test"
}

# where are labels of colors to train
labels_train = {
    "b": "digits_labels_colour_train",
    "c": "colourful_background/labels_train",
    "r": "colourful_background_random/labels_train",
    "gb": "generated_colorful_black_bg/color_labels_train",
    "gc": "generated_colorful_color_bg/color_labels_train",
    "gr": "generated_colorful_color_bg_random/color_labels_train"
}

# where are labels of colors to test
labels_test = {
    "b": "digits_labels_colour_test",
    "c": "colourful_background/labels_test",
    "r": "colourful_background_random/labels_test",
    "gb": "generated_colorful_black_bg/color_labels_test",
    "gc": "generated_colorful_color_bg/color_labels_test",
    "gr": "generated_colorful_color_bg_random/color_labels_test"
}

shape_labels_train = {
    "b": "digits_labels_shapes_train",
    "c": "digits_labels_shapes_train",
    "r": "digits_labels_shapes_train",
    "gb": "generated_colorful_black_bg/shape_labels_train",
    "gc": "generated_colorful_color_bg/shape_labels_train",
    "gr": "generated_colorful_color_bg_random/shape_labels_train"
}
shape_labels_test = {
    "b": "digits_labels_shapes_test",
    "c": "digits_labels_shapes_test",
    "r": "digits_labels_shapes_test",
    "gb": "generated_colorful_black_bg/shape_labels_test",
    "gc": "generated_colorful_color_bg/shape_labels_test",
    "gr": "generated_colorful_color_bg_random/shape_labels_test"
}

black_images = {
    "b": "black_white_digits",
    "c": "black_white_digits",
    "r": "black_white_digits",
    "gb": "generated_colorful_black_bg/digits_bw_train",
    "gc": "generated_colorful_color_bg/digits_bw_train",
    "gr": "generated_colorful_color_bg_random/digits_bw_train"
}
blacks_img = load_pictures(black_images[type_picture])
coloured_train_images = load_pictures(pictures_train[type_picture])  # train images load to array of array
coloured_test_images = load_pictures(pictures_test[type_picture])  # test images load to array of array
#
labels_colour_train, labels_shapes_train, labels_both_train = read_labels(labels_train[type_picture],
                                                                          shape_labels_train[type_picture])
                                                                          # "digits_labels_shapes_train")
# "digits_labels_shapes_train" is hardcoded because labels for shapes are the same for all types of pictures

labels_colour_test, labels_shapes_test, labels_both_test = read_labels(labels_test[type_picture],
                                                                        shape_labels_test[type_picture])
                                                                       # "digits_labels_shapes_test")
# "digits_labels_shapes_train" is hardcoded because labels for shapes are the same for all types of pictures

# make sets of X (expected inputs to neural nets)
x_train_black = np.array(blacks_img)
x_train = np.array(coloured_train_images)
x_test = np.array(coloured_test_images)

# make sets of Y (expected responses from neural nets)
y_set_colour_train = np.array(tf.one_hot(labels_colour_train, NUMBER_OF_COLOURS))
y_set_shape_train = np.array(tf.one_hot(labels_shapes_train, NUMBER_OF_SHAPES))
y_set_both_train = [y_set_colour_train, y_set_shape_train]

y_set_colour_test = np.array(tf.one_hot(labels_colour_test, NUMBER_OF_COLOURS))
y_set_shape_test = np.array(tf.one_hot(labels_shapes_test, NUMBER_OF_SHAPES))
y_set_both_test = [y_set_colour_test, y_set_shape_test]


def train_and_test(num_of_e=1):
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
    print("Both summary:")
    print(model_both.summary())
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
    # print("Shape summary:")
    # print(model_shape.summary())

    #  mix sets
    # x_train_sh, y_set_colour_train_sh, y_set_shape_train_sh = shuffle_sets(coloured_train_images, labels_colour_train,
    #                                                                        labels_shapes_train)
    # print("Without training one network (both features): ==================")
    # print(metrics(model_both.predict(x_test), y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES))
    # result = model_both.evaluate(x_test, y_set_both_test)
    # print(dict(zip(model_both.metrics_names, result)))

    #  train one network (both features)

    start = time.process_time()
    model_both.fit(x_train, [y_set_colour_train, y_set_shape_train], epochs=3)
    time_both = time.process_time() - start

    #  look at how shape_output_accuracy is changed in this model
    # plt.interactive(False)
    # plt.plot(history.history['shape_output_accuracy'])
    # plt.show()

    #  mix sets again

    # x_train_sh, y_set_colour_train_sh, y_set_shape_train_sh = shuffle_sets(coloured_train_images, labels_colour_train,
    #                                                                        labels_shapes_train)
    print("Without training parallel (accuracy):-------------------")
    # print(test_parallel(
    #     model_colour, model_shape, x_test, y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES))  # accuracy
    # result = model_shape.evaluate(x_test, y_set_shape_test)
    # print("Without training (shape model):-------------------")
    # print(dict(zip(model_shape.metrics_names, result)))

    #  train model colour
    start = time.process_time()
    model_colour.fit(x_train, y_set_colour_train, epochs=num_of_e)
    time_c = time.process_time() - start
    #
    # x_train_sh, y_set_colour_train_sh, y_set_shape_train_sh = shuffle_sets(train_images, labels_colour_train,
    #                                                                        labels_shapes_train)
    #  train model shape
    start = time.process_time()
    model_shape.fit(x_train_black, y_set_shape_train, epochs=1) #!!!!!!!!!!! uwaga odkomentuj potem
    model_shape.fit(x_train, y_set_shape_train, epochs=num_of_e)
    time_s = time.process_time() - start

    #  look at how accuracy is changed in this model
    # plt.interactive(False)
    # plt.plot(history.history['accuracy'])
    # plt.show()

    return metrics(model_both.predict(x_test), y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES), test_parallel(
        model_colour, model_shape, x_test, y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES), time_both, time_c+time_s


def test_and_train_times(iterations=5):
    boths = []
    pars = []
    boths_times = []
    pars_times = []
    for i in range(iterations):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(i)
        b, p, b_time, p_time = train_and_test(1)
        boths.append(b)
        pars.append(p)
        boths_times.append(b_time)
        pars_times.append(p_time)
    print(type_picture)
    print("------------ one (both in one)")
    print(sum(boths) / iterations)
    print(boths)
    print("------------ pararel")
    print(sum(pars) / iterations)
    print(pars)
    print("------------ one (both in one) TIME")
    print(sum(boths_times) / iterations)
    print(boths_times)
    print("------------ pararel TIME")
    print(sum(pars_times) / iterations)
    print(pars_times)

#
# both, par = train_and_test()
# print("Accuracy one network (both features): " + str(both))
# print("Accuracy parallel network: " + str(par))

test_and_train_times(5)

# statistic_digits = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0],
#                     6: [0, 0, 0], 7: [0, 0, 0], 8: [0, 0, 0], 9: [0, 0, 0]}
# ind = 0
# for label in labels_shapes_train:
#     statistic_digits[label][int(labels_colour_train[ind])] += 1
#     ind += 1
#
# print(statistic_digits)

def save_coloured_generated_set_of_digits(directory_with_base, dir_for_pic, dir_for_bw_pic, dir_for_color, dir_for_shape, colour_background=False, colour_background_randomly=False):
    for subdir, dirs, files in os.walk(directory_with_base):
        for file in files:

            # label from shape
            ###################
            img = imageio.imread(directory_with_base+"/"+file, pilmode="L")
            shape_of_digit = file.split(".")[0]
            factor = rnd.randint(171, 180)
            for i in range(factor):
                coloured_img, colour_nr = colour_digit(img, colour_background, colour_background_randomly)

                imageio.imwrite(dir_for_bw_pic+ "/generated_" + str(i) + "_" + str(shape_of_digit) + ".png",
                                img)
                imageio.imwrite(dir_for_pic + "/generated_" + str(i) + "_" + str(shape_of_digit) + ".png", np.array(coloured_img))
                save_label_to_file(colour_nr, str(i) + '_' + shape_of_digit, dir_for_color)
                save_label_to_file(shape_of_digit, str(i) + '_' + shape_of_digit, dir_for_shape)


