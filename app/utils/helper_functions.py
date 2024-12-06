import os
import pickle
import sys
import trimesh

from PIL import Image, ImageTk

from src import PredictionModel


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS2
        base_path = sys._MEIPASS2
    except Exception as e:
        print(e)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def load_images():
    # Load images for the tiles
    images: dict[int, (ImageTk.PhotoImage, Image.Image)] = {}

    for i in range(0, 22):
        image = Image.open(resource_path('data/app_images/' + str(i) + ".jpg"))
        # give the image a size according to the menu section
        image = image.resize((50, 50))
        images[i] = (ImageTk.PhotoImage(image), image)

    return images


def load_meshes():
    # Load meshes
    stl_path = resource_path('data/STL/')
    meshes = {}
    # Read all the files in the STL folder and load them into the dictionary.
    # File' name is the key and the mesh object is the value
    for file in os.listdir(stl_path):
        if file.endswith(".STL"):
            meshes[file] = trimesh.load(stl_path + file)

    return meshes


def load_res_bounds():
    # Load upper and lower bounds for the resistance values from the pickle file
    with open(resource_path('data/resistance_bounds.pkl'), 'rb') as f:
        res_ubs_lbs = pickle.load(f)

    print(res_ubs_lbs)
    # print the type of the loaded object
    print(type(res_ubs_lbs))

    return res_ubs_lbs


def load_prediction_model():
    # Load the prediction model from the pickle file
    base_learners_pickle_path = resource_path('drive_data/pickles/base_learner_pickles/')
    meta_learner_pickle_path = resource_path('drive_data/pickles/meta_learner_pickles/')

    prediction_model = PredictionModel(base_learners_pickle_path=base_learners_pickle_path,
                                       meta_learner_pickle_path=meta_learner_pickle_path)

    return prediction_model
