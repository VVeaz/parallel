import generators

from config import BLACK, BLUE, GREEN, RED

if __name__ == "__main__":
    generators.ImageGenerator.generate_all_images(RED, BLUE, "rb")
    generators.ImageGenerator.generate_all_images(RED, GREEN, "rg")
    generators.ImageGenerator.generate_all_images(BLUE, RED, "br")
    generators.ImageGenerator.generate_all_images(BLUE, GREEN, "bg")
    generators.ImageGenerator.generate_all_images(GREEN, RED, "gr")
    generators.ImageGenerator.generate_all_images(GREEN, BLUE, "gb")
    generators.ImageWithFrameGenerator.generate_all_images(RED, BLACK, "frame_rbl")
    generators.ImageWithFrameGenerator.generate_all_images(GREEN, BLACK, "frame_gbl")
    generators.ImageWithFrameGenerator.generate_all_images(BLUE, BLACK, "frame_bbl")
