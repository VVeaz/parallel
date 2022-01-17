import os
from PIL import Image
import tensorflow as tf

from tensorflow.python.keras import backend as K


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


def metrics(y_predict_prob, y_true, number_of_colours, number_of_shaps):  # NOW its only accuracy
    equal = 0
    all_examples = len(y_true[0])
    y_predict = [tf.one_hot(K.argmax(y_predict_prob[0], axis=-1), number_of_colours),
                 tf.one_hot(K.argmax(y_predict_prob[1], axis=-1), number_of_shaps)]
    for i in range(all_examples):
        colour_bool = y_predict[0][i] == y_true[0][i]
        shape_bool = y_predict[1][i] == y_true[1][i]
        if tf.reduce_all(colour_bool) and tf.reduce_all(shape_bool):
            equal += 1
    return equal / all_examples


def test_parallel(model_for_colour, model_for_shape, x_test, y_test, number_of_colours, number_of_shaps):
    y_colour_predict = model_for_colour.predict(x_test)  # predict colour
    y_shape_predict = model_for_shape.predict(x_test)  # predict shape
    y_predict = [y_colour_predict, y_shape_predict]  # here will be cartesian product of result from colour and shape
    metric = metrics(y_predict, y_test, number_of_colours, number_of_shaps)  # TODO better metrics
    return metric
