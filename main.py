import sys
from typing import List, Tuple

from PySide6.QtCore import Qt, QSize, QDir
from PySide6.QtGui import QImage, QColor

RED = Qt.GlobalColor.red
GREEN = Qt.GlobalColor.green
BLUE = Qt.GlobalColor.blue
BLACK = Qt.GlobalColor.black

IMAGE_HEIGHT = 3
IMAGE_WIDTH = 3

# To make a frame around real image it's possible to turn off some pixels
# Mark pixel as 0 to turn it off. By default all pixels are turned on.
OFF_PIXELS = (1,) * (IMAGE_WIDTH * IMAGE_WIDTH)


def create_image(pixels: Tuple, fg_colour: QColor, bg_colour: QColor, save_dir: str):
    filename = "".join(map(str, pixels))

    colour_pixels = list(map(lambda x: fg_colour if x[0] and x[1] else bg_colour, zip(pixels, OFF_PIXELS)))

    image = QImage(QSize(IMAGE_WIDTH, IMAGE_HEIGHT), QImage.Format.Format_RGB32)

    for i in range(0, IMAGE_WIDTH * IMAGE_HEIGHT):
        image.setPixelColor(i % IMAGE_WIDTH, i // IMAGE_WIDTH, colour_pixels[i])

    image.save(fr"images/{save_dir}/{filename}.png", "png", 100)


def generate_all_images(fg_colour: QColor, bg_colour: QColor, save_dir: str):
    create_dir(save_dir)

    fixed_length = IMAGE_HEIGHT * IMAGE_WIDTH
    for i in range(2 ** (IMAGE_HEIGHT * IMAGE_WIDTH)):
        pixels = f"{i:0{fixed_length}b}"  # make binary number in string with length = fixed_length
        create_image(tuple(map(lambda x: int(x), pixels)),  # change string into list of ints
                     fg_colour,
                     bg_colour,
                     save_dir)


def create_dir(save_dir: str):
    current = QDir.current()

    # It may fail but we do not care, it could already exist
    current.mkdir("images")
    current.cd("images")

    if not current.mkdir(save_dir):
        sys.stderr.write(f"Unable to create dir '{save_dir}' in {current.absolutePath()}\n")


if __name__ == "__main__":
    # generate_all_images(RED, BLUE, "rb")
    # generate_all_images(RED, GREEN, "rg")
    # generate_all_images(BLUE, RED, "br")
    # generate_all_images(BLUE, GREEN, "bg")
    # generate_all_images(GREEN, RED, "gr")
    # generate_all_images(GREEN, BLUE, "gb")
    generate_all_images(RED, BLACK, "rbl")
    generate_all_images(GREEN, BLACK, "gbl")
    generate_all_images(BLUE, BLACK, "bbl")
