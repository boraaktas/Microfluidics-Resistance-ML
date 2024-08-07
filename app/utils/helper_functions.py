import os
import sys

from PIL import Image, ImageTk


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception as e:
        print(e)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def load_images():
    # Load images for the tiles
    images = []

    for i in range(0, 22):
        image = Image.open(resource_path('data/app_images/' + str(i) + ".jpg"))
        # give the image a size according to the menu section
        image = image.resize((50, 50))
        images.append((i, ImageTk.PhotoImage(image), image))

    return images
