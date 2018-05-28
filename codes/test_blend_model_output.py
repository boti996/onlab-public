# TODO: megcsinálni

import pickle

import keras.losses
import numpy as np
from keras.models import load_model

import helpers as helper
import my_losses as mylosses


size = (320, 256)
images_path = 'datas/images_mlnd/image/full_CNN_train.p'
images = pickle.load(open(images_path, "rb"))
images = images[:300]
images = helper.resize_images(images, size)

images = np.array(images) / 255

keras.losses.mean_squared_error_coeff = mylosses.mean_squared_error_coeff
model = load_model("models/mlnd_full.h5")

helper.blend_images_polynom(images=images, model=model, folder_url='../datas/images_mlnd/output/')