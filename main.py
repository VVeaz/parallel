import sys
from itertools import permutations
from typing import List, Tuple

from PySide6.QtCore import Qt, QSize, QDir
from PySide6.QtGui import QImage, QColor

RED = Qt.GlobalColor.red
GREEN = Qt.GlobalColor.green
BLUE = Qt.GlobalColor.blue


def create_image(pixels: Tuple, fg_colour: QColor, bg_colour: QColor, save_dir: str):
    filename = "".join(map(str, pixels))
    colour_pixels = list(map(lambda x: fg_colour if x == 1 else bg_colour, pixels))

    image = QImage(QSize(3, 3), QImage.Format.Format_RGB32)

    for i in range(0, 9):
        image.setPixelColor(i % 3, i // 3, colour_pixels[i])

    image.save(fr"{save_dir}/{filename}.png", "png")


def create_all_possible_permutations_and_gen_image(data: List[int], fg_colour: QColor, bg_colour: QColor,
                                                   save_dir: str):
    unique = set()
    for p in permutations(data):
        unique.add(p)

    for permutation in unique:
        create_image(permutation, fg_colour, bg_colour, save_dir)


def generate_all_images(fg_colour: QColor, bg_colour: QColor, save_dir: str):
    create_image((0, 0, 0, 0, 0, 0, 0, 0, 0), fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 0, 0, 0, 0, 0, 0, 0, 0], fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 1, 0, 0, 0, 0, 0, 0, 0], fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 1, 1, 0, 0, 0, 0, 0, 0], fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 1, 1, 1, 0, 0, 0, 0, 0], fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 1, 1, 1, 1, 0, 0, 0, 0], fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 1, 1, 1, 1, 1, 0, 0, 0], fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 1, 1, 1, 1, 1, 1, 0, 0], fg_colour, bg_colour, save_dir)
    create_all_possible_permutations_and_gen_image([1, 1, 1, 1, 1, 1, 1, 1, 0], fg_colour, bg_colour, save_dir)
    create_image((1, 1, 1, 1, 1, 1, 1, 1, 1), fg_colour, bg_colour, save_dir)


def create_dirs():
    current = QDir.current()

    dirs_to_create = ["rb", "rg", "br", "bg", "gb", "gr"]

    for directory in dirs_to_create:
        if not current.mkdir(fr"images/{directory}"):
            sys.stderr.write(f"Unable to create dir '{directory}' in {current.absolutePath()}\n")


if __name__ == "__main__":
    create_dirs()

    generate_all_images(RED, BLUE, "rb")
    generate_all_images(RED, GREEN, "rg")
    generate_all_images(BLUE, RED, "br")
    generate_all_images(BLUE, GREEN, "bg")
    generate_all_images(GREEN, RED, "gr")
    generate_all_images(GREEN, BLUE, "gb")
