import os
import pickle

import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import codes.my_helper as helper
import codes.my_models as my_models
import codes.my_losses as my_losses

import model as my_models_old

def main():
    # TODO: Camvid tanítás, többféle lr, batch size, loss, 100 epoch, majd a legjobb kombót viszi tovább 1000 epochhal
    # DATA

    # Train images: read from images folder, resize, normalize to 0..1 range
    data_path = '../datas/'
    images = helper.read_images(data_path + 'images_camvid/image/')

    size = (320, 256)
    images = helper.resize_images(images, size)

    images = np.array(images) / 255

    # Train labels: read transformed labels from file if exists
    # else read images from folder,
    # remove unnecessary label-information, resize, transform, and save transformed labels to file
    labels_path = data_path + 'labels_transformed_camvid.p'

    if os.path.exists(labels_path):
        labels = pickle.load(open(labels_path, "rb"))
    else:
        labels = helper.read_images(data_path + 'images_camvid/label/')

        labels = helper.clear_label_colors(labels)

        labels = helper.resize_images(labels, size)

        labels = np.array(labels)
        classes = [[192, 0, 128]]
        labels = helper.rgb_to_classes(labels, classes)     # TODO: rgb_to_binary!

        pickle.dump(labels, open(labels_path, "wb"))

    # Shuffle dateset, then create training- and validation arrays
    img_train, img_val, label_train, label_val = train_test_split(images, labels, test_size=0.15,
                                                                  shuffle=True, random_state=helper.random_state)
    # helper.blend_images(images=img_val, labels=label_val, folder_url=data_path + 'images_camvid/output/')

    # MODEL

    # Main model parameters
    batch_size = 20
    epochs = 100
    input_shape = img_train.shape[1:]
    dropout_rate = 0.2
    pool_size = (2, 2)
    learning_rate = 0.001

    # Load model structure
    model = my_models.get_model_segm(input_shape, pool_size, dropout_rate)

    # TODO: custom loss!
    # Initialize model + print model
    model.compile(optimizer=Adam(lr=learning_rate), loss=categorical_crossentropy)  # my_losses.weighted_binary_crossentropy

    # model_path = '../models/camvid_weights.400.h5'
    # model.load_weights(model_path, by_name=True)

    my_models.train_model(model, img_train, img_val, label_train, label_val, batch_size, epochs,
                          log_path='../logs/camvid', save_path='../models/camvid', out_path=data_path + 'images_camvid/output/', datagen=True)


main()
