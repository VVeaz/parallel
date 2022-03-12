import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense
import time
from tensorflow.python.keras import backend as K
from matplotlib import pyplot

# black background and frame - 9241
# black background - 9405
# colourful background - 10000
# read_labels("new_labels_colours", "new_labels_shapes") # colourful background
# load_pictures("good_examples")
# read_labels("new_labels_colours_bl", "new_labels_shapes_bl") # black background
# load_pictures("good_examples_bl")
# read_labels("labels_colours_frame", "labels_shapes_frame") # frame background
# load_pictures("good_examples_frame")
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


def custom_loss(y_pred, y_true):
    Loss = 0
    y_pred1 = y_pred[0]
    y_pred2 = y_pred[1]
    y_true1 = y_true[0]
    y_true2 = y_true[1]


    def loss1(y_true1, y_pred1):
        return np.square(np.subtract(y_true1, y_pred1)).mean()

    def loss2(y_true2, y_pred2):
        return np.square(np.subtract(y_true2, y_pred2)).mean()

    def finalloss(y_pred1, y_true1, y_pred2, y_true2):
        Loss = loss1(y_pred1, y_true1) + loss2(y_pred2, y_true2)
        if (y_pred1 == y_true1 and y_pred2 == y_true2):
            return (0)
        elif (y_pred1 == y_true1 and y_pred2 != y_true2):
            return (0.5 * Loss)
        elif (y_pred1 != y_true1 and y_pred2 == y_true2):
            return (0.5 * Loss)
        else:
            return (Loss)

    return finalloss(y_pred1, y_true1, y_pred2, y_true2)


def read_labels(dir_colour, dir_shape):
    # read colour labels from files to array >>labels_from_color<<
    root_dir = dir_colour
    labels_col = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):
            with open(root_dir + "/" + file, "r") as label_color:
                print("=========")
                print(file)
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
                    # example.append(rgb2int(pixels[i, j]))
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


def m_accuracy(y_predict_prob_c, y_predict_prob_s, y_true_c, y_true_s):  # NOW its only accuracy
    y_true = [y_true_c, y_true_s]
    y_predict_prob = [y_predict_prob_c, y_predict_prob_s]
    equal = 0
    all_examples = len(y_true[0])
    y_predict = [tf.one_hot(K.argmax(y_predict_prob[0], axis=-1), NUMBER_OF_COLOURS),
                 tf.one_hot(K.argmax(y_predict_prob[1], axis=-1), NUMBER_OF_SHAPES)]
    for i in range(all_examples):
        # print(y_predict[0][i])
        # print(y_predict[1][i])
        # print(y_true[0][i])
        # print(y_true[1][i])
        colour_bool = y_predict[0][i] == y_true[0][i]
        shape_bool = y_predict[1][i] == y_true[1][i]
        # print(colour_bool)
        # print(shape_bool)
        if tf.reduce_all(colour_bool) and tf.reduce_all(shape_bool):
            equal += 1
    return equal / all_examples


def metrics(y_predict_prob, y_true):  # NOW its only accuracy
    equal = 0
    all_examples = len(y_true[0])
    y_predict = [tf.one_hot(K.argmax(y_predict_prob[0], axis=-1), NUMBER_OF_COLOURS),
                 tf.one_hot(K.argmax(y_predict_prob[1], axis=-1), NUMBER_OF_SHAPES)]
    for i in range(all_examples):
        # print(y_predict[0][i])
        # print(y_predict[1][i])
        # print(y_true[0][i])
        # print(y_true[1][i])
        colour_bool = y_predict[0][i] == y_true[0][i]
        shape_bool = y_predict[1][i] == y_true[1][i]
        # print(colour_bool)
        # print(shape_bool)
        if tf.reduce_all(colour_bool) and tf.reduce_all(shape_bool):
            equal += 1
    return equal / all_examples


def test_parallel(model_for_colour, model_for_shape, x_test, y_test):
    y_colour_predict = model_for_colour.predict(x_test)  # predict colour
    y_shape_predict = model_for_shape.predict(x_test)  # predict shape
    y_predict = [y_colour_predict, y_shape_predict]  # here will be cartesian product of result from colour and shape
    metric = metrics(y_predict, y_test)
    print("Parallel")
    print(metric)  # TODO better metrics
    return metric
    # for i in range(len(y_test[0])):
    #     print(y_colour_predict[i])
    #     print(y_shape_predict[i])
    #     print(y_test[0][i])
    #     print(y_test[1][i])
    #     # print(y_test[0][i] == y_colour_predict[i])
    #     # print(y_test[i] == y_predict_np[i])
    #     print("-------")


# =========================================================== READ DATA
labels_from_color, labels_from_shapes, labels_from_both = read_labels(variables_in_type["colours"],
                                                                      variables_in_type["shapes"])
# read_labels("new_labels_colours", "new_labels_shapes") # colourful background
# load_pictures("good_examples")
# read_labels("new_labels_colours_bl", "new_labels_shapes_bl") # black background
# load_pictures("good_examples_bl")
# read_labels("labels_colours_frame", "labels_shapes_frame") # frame background
# load_pictures("good_examples_frame")

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
    # --- training
    start = time.process_time()
    history = model_both.fit(x_train_set, y_train_set_both, epochs=3)
    print(history.history.keys())
    print("-----------------------------------")
    print(history.history['colour_output_metrics'])
    print("-----------------------------------")

    print("-----------------------------------")
    print(history.history['shape_output_metrics'])
    print("-----------------------------------")
    #
    pyplot.plot(history.history['colour_output_accuracy'])
    pyplot.show()
    pyplot.plot(history.history['shape_output_metrics'])
    pyplot.show()

    time_of_learning_one = time.process_time() - start
    start = time.process_time()
    model_colour.fit(x_train_set, y_train_set_colour, epochs=3)
    time_c = time.process_time() - start
    start = time.process_time()
    model_shape.fit(x_train_set, y_train_set_shape, epochs=3)
    time_s = time.process_time() - start
    time_of_learning_parallel = time_c+time_s

    # result = model_both .evaluate(x_test_set, y_test_set_both)
    # auto. eval. of the results of the network trained for both features
    # print(dict(zip(model_both.metrics_names, result)))
    print("+++++++++++")
    print(metrics(model_both.predict(x_test_set), y_test_set_both))
    test_parallel(model_colour, model_shape, x_test_set,
                  y_test_set_both)  # manual testing network consisting of two nets
    print("+++++++++++")
    aaa = model_both.predict(np.array([x_train_set[0]]))
    print(np.argmax(aaa[0], axis=-1))
    print(tf.one_hot(np.argmax(aaa[0], axis=-1), NUMBER_OF_COLOURS))
    print(K.argmax(aaa[0], axis=-1))
    print(tf.one_hot(K.argmax(aaa[0], axis=-1), NUMBER_OF_COLOURS))
    print(K.one_hot(K.argmax(aaa[0], axis=-1), NUMBER_OF_COLOURS))
    return metrics(model_both.predict(x_test_set), y_test_set_both), test_parallel(model_colour, model_shape,
                                                                                   x_test_set,
                                                                                   y_test_set_both),\
           time_of_learning_one, time_of_learning_parallel, time_c,  time_s
    # print(x_train_set[0])
    # print(metrics_single(model_colour.predict(x_test_set), y_test_set_colour, NUMBER_OF_COLOURS))
    # print(metrics_single(model_shape.predict(x_test_set), y_test_set_shape, NUMBER_OF_SHAPES))
    #
    # print(model_shape.predict(np.array([x_train_set[0]])))
    # print(y_test_set_shape[0])
    #
    # print(model_shape.predict(np.array([x_train_set[1]])))
    # print(y_test_set_shape[1])
    #
    # print(model_shape.predict(np.array([x_train_set[2]])))
    # print(y_test_set_shape[2])
    #
    # print(model_shape.predict(np.array([x_train_set[0]])))
    # print(y_test_set_shape[3])
    #
    # print(model_shape.predict(np.array([x_train_set[1]])))
    # print(y_test_set_shape[4])
    #
    # print(model_shape.predict(np.array([x_train_set[2]])))
    # print(y_test_set_shape[5])
    #
    # for i in range(15):
    #     print("-----------------------------")
    #     print(y_train_set_shape[i])
    #     print(labels_from_shapes[i])


# ones = []
# parallels = []
# ones_times = []
# parallels_times = []
# c_times = []
# s_times = []
# iterations = 50
# for i in range(iterations):
#     one, parallel, time_one, time_parallel, time_colour, time_shape = train_and_test()
#     ones.append(one)
#     parallels.append(parallel)
#     ones_times.append(time_one)
#     c_times.append(time_colour)
#     s_times.append(time_shape)
#     parallels_times.append(time_parallel)
# print(ones)
# print(parallels)
# print(sum(ones) / iterations)
# print(sum(parallels) / iterations)
# print(ones_times)
# print(parallels_times)
# print(sum(ones_times) / iterations)
# print(sum(parallels_times) / iterations)
# print(sum(c_times) / iterations)
# print(sum(s_times) / iterations)
