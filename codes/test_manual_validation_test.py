import os

import numpy as np
from keras.models import load_model

import codes.my_helper as helper


def main():
    # Set number of test image folders
    no_test_folders = 3
    size = (320, 256)
    models_path = '../models/'

    for i in range(0, no_test_folders):
        # Prepare input images
        image_path = '../datas/test/test_images' + str(i)
        images = helper.read_images(image_path+'/image/')
        images = helper.resize_images(images, size)
        images = np.array(images) / 255

        # Load models
        model_names = np.array(['roma_full.freeze.001.500.h5']) # , 'camvid_full.500.h5', 'roma_full.001.500.h5', 'roma_full.0001.500.h5'])
        for n in range(0, len(model_names)):

            model_name = model_names[n]
            model = load_model(models_path + model_name)
            # blend output images
            directory = image_path+'/output/' + model_name + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            helper.blend_images(images=images, model=model, folder_url=directory)
            print()
            print()


main()
