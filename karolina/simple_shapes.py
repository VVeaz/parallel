import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense
import time
from tensorflow.python.keras import backend as K

variables_types_data = {
    "c": {  # colourful
        "colours": "new_labels_colours",
        "shapes": "new_labels_shapes",
        "pictures": "good_examples",
        "side_length": 3,
        "data_size": 10000
    },
    "b": {  # black
        "colours": "new_labels_colours_bl",
        "shapes": "new_labels_shapes_bl",
        "pictures": "good_examples_bl",
        "side_length": 3,
        "data_size": 9405
    },
    "f": {  # frame
        "colours": "labels_colours_frame",
        "shapes": "labels_shapes_frame",
        "pictures": "good_examples_frame",
        "side_length": 5,
        "data_size": 9421
    }
}
type_of_data = "b"
variables_in_type = variables_types_data[type_of_data]
DATA_SET_SIZE = variables_in_type["data_size"]
SPLIT = int(DATA_SET_SIZE * 0.8)
NUMBER_OF_COLOURS = 3
NUMBER_OF_SHAPES = 3
SIDE_LENGTH = variables_in_type["side_length"]


def read_labels(dir_colour, dir_shape):
    # read colour labels from files to array >>labels_from_color<<
    root_dir = dir_colour
    labels_col = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):
            with open(root_dir + "/" + file, "r") as label_color:
                labels_col.append(int(label_color.readline()))

    # read shape labels from files to array >>labels_from_shapes<<
    # make array with shape&&colour >>labels_from_both<<
    root_dir = dir_shape
    labels_shap = []
    labels_both = []
    i = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):
            with open(root_dir + "/" + file, "r") as label_shape:
                labels_shap.append(int(label_shape.readline()))
                labels_both.append([labels_col[i], labels_shap[i]])
                i += 1
    return labels_col, labels_shap, labels_both


def load_pictures(root_dir):
    x = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):
            img = Image.open(root_dir + "/" + file).convert("RGB")
            pixels = img.load()
            example_two_dim = []
            img_width, img_height = img.size  # SIDE_LENGTH x SIDE_LENGTH
            for i in range(img_height):
                example = []
                for j in range(img_width):
                    example.append(pixels[i, j])
                example_two_dim.append(example)
            x.append(example_two_dim)
    return x


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
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3)),
        tf.keras.layers.Dense(SIDE_LENGTH * SIDE_LENGTH, activation='relu'),
        tf.keras.layers.Dense(SIDE_LENGTH, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUMBER_OF_COLOURS, activation='softmax')
    ])


def build_shape_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(SIDE_LENGTH, SIDE_LENGTH, 3)),
        tf.keras.layers.Dense(SIDE_LENGTH * SIDE_LENGTH, activation='relu'),
        tf.keras.layers.Dense(SIDE_LENGTH, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUMBER_OF_SHAPES, activation='softmax')
    ])


def metrics_single(y_predict_prob, y_true, size):  # NOW its only accuracy
    equal = 0
    all_examples = len(y_true)
    y_predict = tf.one_hot(np.argmax(y_predict_prob, axis=-1), size)
    for i in range(all_examples):
        _bool = y_predict[i] == y_true[i]
        if tf.reduce_all(_bool):
            equal += 1
    return equal / all_examples


def m_accuracy(y_predict_prob_c, y_predict_prob_s, y_true_c, y_true_s):
    y_true = [y_true_c, y_true_s]
    y_predict_prob = [y_predict_prob_c, y_predict_prob_s]
    equal = 0
    all_examples = len(y_true[0])
    y_predict = [tf.one_hot(K.argmax(y_predict_prob[0], axis=-1), NUMBER_OF_COLOURS),
                 tf.one_hot(K.argmax(y_predict_prob[1], axis=-1), NUMBER_OF_SHAPES)]
    for i in range(all_examples):
        colour_bool = y_predict[0][i] == y_true[0][i]
        shape_bool = y_predict[1][i] == y_true[1][i]
        if tf.reduce_all(colour_bool) and tf.reduce_all(shape_bool):
            equal += 1
    return equal / all_examples


def metrics(y_predict_prob, y_true):
    equal = 0
    all_examples = len(y_true[0])
    y_predict = [tf.one_hot(K.argmax(y_predict_prob[0], axis=-1), NUMBER_OF_COLOURS),
                 tf.one_hot(K.argmax(y_predict_prob[1], axis=-1), NUMBER_OF_SHAPES)]
    for i in range(all_examples):
        colour_bool = y_predict[0][i] == y_true[0][i]
        shape_bool = y_predict[1][i] == y_true[1][i]
        if tf.reduce_all(colour_bool) and tf.reduce_all(shape_bool):
            equal += 1
    return equal / all_examples


def test_parallel(model_for_colour, model_for_shape, x_test, y_test):
    y_colour_predict = model_for_colour.predict(x_test)  # predict colour
    y_shape_predict = model_for_shape.predict(x_test)  # predict shape
    y_predict = [y_colour_predict, y_shape_predict]  # here will be cartesian product of result from colour and shape
    metric = metrics(y_predict, y_test)
    print("Parallel")
    print(metric)
    return metric


def train_and_test():
    # ======================================
    # MODELS
    # --------- a model for recognizing color and shape (which will say, for example, "0, 1" - red circle)
    model_both = build_model_both()
    # --------- a color recognition model (which will say, for example, 0 -red)
    model_colour = build_colour_model()
    # --------- a model for recognizing a shape (which will say, for example, 1 - circle)
    model_shape = build_shape_model()
    # ======================================
    # --- compiling
    model_both.compile(optimizer='adam',
                       loss=
                       {'colour_output': tf.keras.losses.CategoricalCrossentropy(),
                        'shape_output': tf.keras.losses.CategoricalCrossentropy()
                        }
                       ,
                       metrics=['accuracy', tf.metrics.Precision(), metrics])
    model_colour.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(),
                         metrics=['accuracy', tf.metrics.Precision()])
    model_shape.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy', tf.metrics.Precision()])
    # --- training one
    start = time.process_time()
    model_both.fit(x_train_set, y_train_set_both, epochs=3)
    #

    time_of_learning_one = time.process_time() - start
    # --- training both
    start = time.process_time()
    model_colour.fit(x_train_set, y_train_set_colour, epochs=3)
    time_c = time.process_time() - start
    start = time.process_time()
    model_shape.fit(x_train_set, y_train_set_shape, epochs=3)
    #

    time_s = time.process_time() - start
    time_of_learning_parallel = time_c + time_s

    return metrics(model_both.predict(x_test_set), y_test_set_both), test_parallel(model_colour, model_shape,
                                                                                   x_test_set,
                                                                                   y_test_set_both), \
           time_of_learning_one, time_of_learning_parallel, time_c, time_s


def test_and_train_times(iterations=5):
    boths = []
    pars = []
    boths_times = []
    pars_times = []
    for i in range(iterations):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(i)
        b, p, b_time, p_time, c_time, s_time = train_and_test()
        boths.append(b)
        pars.append(p)
        boths_times.append(b_time)
        pars_times.append(p_time)
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


# =========================================================== READ DATA
labels_from_color, labels_from_shapes, labels_from_both = read_labels(variables_in_type["colours"],
                                                                      variables_in_type["shapes"])

# make set of X (inputs to neural nets)
x_set_array = load_pictures(variables_in_type["pictures"])

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

y_test_set_colour = y_set_colour[SPLIT:]
y_test_set_shape = y_set_shape[SPLIT:]
y_test_set_both = [y_test_set_colour, y_test_set_shape]

# ===========================================================


test_and_train_times(5)
