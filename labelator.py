"""
This is LaBeLaToR
run in a folder one higher than "ALL_ONCE"
"""

import os
from shutil import copyfile


def copy_from_folders_to_one_folder(root_dir, new_dir):
    for subdir1, dirs1, files1 in os.walk(root_dir):
        for dire in dirs1:  # rg, gb,.. etc.
            for subdir, dirs, files in os.walk(os.path.join(root_dir, dire)):
                for file in files:
                    copyfile(os.path.join(subdir, file), os.path.join(new_dir, dire + file))
                    # "ALL_ONCE/bg/010101010.png" -> one_folder/bg010101010.png


def make_labels(root_dir, label_dir, prefix, shapes_array, colour="x"):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:

            # label from shape
            ###################
            clean_file_name = file.split(".")[0]  # without extension, example: bg010101010
            foreground_shape = clean_file_name[2:]  # example: 010101010

            label_shape_filename = os.path.join(label_dir + "_shapes", prefix + clean_file_name + ".txt")
            os.makedirs(os.path.dirname(label_shape_filename), exist_ok=True)

            with open(label_shape_filename, "w") as file_shape:
                label = "-1"
                for i in range(len(array_of_shapes)):
                    if foreground_shape == array_of_shapes[i]:
                        label = str(i)
                        break
                file_shape.write(label)

            # label from colour (if it is necessary)
            ########################################

            if colour != "x":
                foreground_colour = clean_file_name[0]  # example: b

                label_colour_filename = os.path.join(label_dir + "_colours", prefix + file.split(".")[0] + ".txt")
                os.makedirs(os.path.dirname(label_colour_filename), exist_ok=True)

                with open(label_colour_filename, "w") as file_colour:
                    if foreground_colour == colour:
                        file_colour.write("1")
                    else:
                        file_colour.write("0")


plus = "010" \
       "111" \
       "010"
circle = "111" \
         "101" \
         "111"
ex = "100" \
     "010" \
     "001"
array_of_shapes = [plus, circle, ex]
colour_being_searching_for = "r"

copy_from_folders_to_one_folder("ALL_ONCE", "one_folder")
make_labels("one_folder", "labels", "label_", array_of_shapes, colour_being_searching_for)
