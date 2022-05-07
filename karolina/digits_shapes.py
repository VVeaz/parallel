import tensorflow as tf
import numpy as np
import time
from own_functions import load_pictures, read_labels, test_parallel, metrics
from tensorflow.python.keras import Model, Input
from tensorflow.keras.layers import Dense

NUMBER_OF_COLOURS = 3
NUMBER_OF_SHAPES = 10
SIDE_LENGTH = 28  # number of pixels on one side of picture


def build_model_both():
    input_layer = Input(shape=(SIDE_LENGTH, SIDE_LENGTH, 3))
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255,
                                                                         input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3))(
        input_layer)
    layer_flatten = tf.keras.layers.Flatten()(normalization)
    layer_1 = Dense(SIDE_LENGTH * SIDE_LENGTH, activation="relu")(layer_flatten)

    layer_2 = Dense(SIDE_LENGTH, activation="relu")(layer_1)
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
    y2_output = Dense(NUMBER_OF_SHAPES, activation='softmax', name='shape_output')(layer_2)
    return Model(inputs=input_layer, outputs=[y2_output])


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

    #  train one network (both features)

    start = time.process_time()
    model_both.fit(x_train, [y_set_colour_train, y_set_shape_train], epochs=3)
    time_both = time.process_time() - start

    #  train model colour
    start = time.process_time()
    model_colour.fit(x_train, y_set_colour_train, epochs=num_of_e)
    time_c = time.process_time() - start
    #                                                                        labels_shapes_train)
    #  train model shape
    start = time.process_time()
    model_shape.fit(x_train_black, y_set_shape_train, epochs=1)
    model_shape.fit(x_train, y_set_shape_train, epochs=num_of_e)
    time_s = time.process_time() - start

    return metrics(model_both.predict(x_test), y_set_both_test, NUMBER_OF_COLOURS, NUMBER_OF_SHAPES), test_parallel(
        model_colour, model_shape, x_test, y_set_both_test, NUMBER_OF_COLOURS,
        NUMBER_OF_SHAPES), time_both, time_c + time_s


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

test_and_train_times(5)
