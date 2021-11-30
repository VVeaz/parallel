import sys
from typing import Tuple

from PySide6.QtCore import QDir, QSize
from PySide6.QtGui import QColor, QImage

from config import *


class ImageGenerator:

    @staticmethod
    def create_image(pixels: Tuple, fg_colour: QColor, bg_colour: QColor, save_dir: str):
        filename = "".join(map(str, pixels))

        colour_pixels = list(map(lambda x: fg_colour if x == 1 else bg_colour, pixels))

        image = QImage(QSize(IMAGE_WIDTH, IMAGE_HEIGHT), QImage.Format.Format_RGB32)
        image.fill(bg_colour)  # Set background colour (whole picture is painted by given colour)

        for i in range(0, IMAGE_WIDTH * IMAGE_HEIGHT):
            if colour_pixels[i] == BLACK:
                continue
            image.setPixelColor(i % IMAGE_WIDTH, i // IMAGE_WIDTH, colour_pixels[i])

        if not image.save(fr"{PARENT_FOLDER_PATH}/{save_dir}/{filename}.png", "png", 100):
            sys.stderr.write(f"Unable to save picture: {filename}.png\n")

    @classmethod
    def generate_all_images(cls, fg_colour: QColor, bg_colour: QColor, save_dir: str):
        cls.create_dir(save_dir)

        fixed_length = IMAGE_HEIGHT * IMAGE_WIDTH
        for i in range(2 ** (IMAGE_HEIGHT * IMAGE_WIDTH)):
            pixels = f"{i:0{fixed_length}b}"  # make binary number in string with length = fixed_length
            cls.create_image(tuple(map(int, pixels)),  # change string into list of ints
                             fg_colour,
                             bg_colour,
                             save_dir)

    @staticmethod
    def create_dir(save_dir: str):
        current = QDir.current()

        # It may fail but we do not care, it could already exist
        current.mkdir("images")
        current.cd("images")

        if not current.mkdir(save_dir):
            sys.stderr.write(f"Unable to create dir '{save_dir}' in {current.absolutePath()}\n")


class ImageWithFrameGenerator(ImageGenerator):
    IMAGE_WIDTH_WITH_FRAME = IMAGE_WIDTH + 2
    IMAGE_HEIGHT_WITH_FRAME = IMAGE_HEIGHT + 2

    @staticmethod
    def create_image(pixels: Tuple, fg_colour: QColor, bg_colour: QColor, save_dir: str):
        filename = "".join(map(str, pixels))

        # First row + first pixel in second low
        pixels_with_frame = [0, ] * (ImageWithFrameGenerator.IMAGE_WIDTH_WITH_FRAME + 1)
        for i in range(IMAGE_WIDTH):
            # Internal pixels + last pixel from current row and first pixel in next row
            pixels_with_frame += list(map(int, pixels[i * IMAGE_HEIGHT:(i + 1) * IMAGE_HEIGHT])) + [0, ] * 2
        # Last pixel in last by one row and last row
        pixels_with_frame += [0, ] * (ImageWithFrameGenerator.IMAGE_WIDTH_WITH_FRAME + 1)

        colour_pixels = list(map(lambda x: fg_colour if x == 1 else bg_colour, pixels_with_frame))

        image = QImage(QSize(ImageWithFrameGenerator.IMAGE_WIDTH_WITH_FRAME,
                             ImageWithFrameGenerator.IMAGE_HEIGHT_WITH_FRAME),
                       QImage.Format.Format_RGB32)
        image.fill(bg_colour)  # Set background colour (whole picture is painted by given colour)

        num_of_pixels = ImageWithFrameGenerator.IMAGE_WIDTH_WITH_FRAME * ImageWithFrameGenerator.IMAGE_HEIGHT_WITH_FRAME
        for i in range(0, num_of_pixels):
            if colour_pixels[i] == bg_colour:
                continue
            image.setPixelColor(i % ImageWithFrameGenerator.IMAGE_WIDTH_WITH_FRAME,  # pixel in row
                                i // ImageWithFrameGenerator.IMAGE_WIDTH_WITH_FRAME,  # row number
                                colour_pixels[i])

        if not image.save(fr"{PARENT_FOLDER_PATH}/{save_dir}/{filename}.png", "png", 100):
            sys.stderr.write(f"Unable to save picture: {filename}.png\n")
