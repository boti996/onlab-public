import os
import pickle

import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import codes.my_helper as helper
import codes.my_models as my_models


def main():
    
    # DATA

    # Train images: read from images folder, resize, normalize to 0..1 range
    data_path = 'datas/'
    images_camvid = helper.read_images(data_path + 'images_camvid/image/')
    size = (320, 256)
    images_camvid = helper.resize_images(images_camvid, size)
    images_camvid = np.array(images_camvid) / 255

    images_roma = helper.read_images(data_path + 'images_roma/image/')
    images_roma = helper.resize_images(images_roma, size)
    images_roma / np.array(images_roma) / 255

    images = np.concatenate(images_camvid, images_roma)

    # Train labels: read transformed labels from file if exists
    # else read images from folder,
    # remove unnecessary label-information, resize, transform, and save transformed labels to file
    labels_path_camvid = data_path + 'labels_transformed_camvid.p'

    if os.path.exists(labels_path_camvid):
        labels_camvid = pickle.load(open(labels_path_camvid, "rb"))
    else:
        labels_camvid = helper.read_images(data_path + 'images_camvid/label/')
        labels_camvid = helper.clear_label_colors(labels_camvid)
        labels_camvid = helper.resize_images(labels_camvid, size)
        labels_camvid = np.array(labels_camvid)
        classes = [[192, 0, 128]]
        labels_camvid = helper.rgb_to_classes(labels_camvid, classes)     # TODO: rgb_to_binary!
        pickle.dump(labels_camvid, open(labels_path_camvid, "wb"))

    labels_path_roma = data_path + 'labels_transformed_roma.p'

    if os.path.exists(labels_path_roma):
        labels_roma = pickle.load(open(labels_path_roma, "rb"))
    else:
        labels_roma = helper.read_images(data_path + 'images_roma/label/')

        labels_roma = helper.resize_images(labels_roma, size)

        labels_roma = np.array(labels_roma)
        classes = [[255, 255, 255]]
        labels_roma = helper.rgb_to_classes(labels_roma, classes)     # TODO: rgb_to_binary!

        pickle.dump(labels_roma, open(labels_path_roma, "wb"))

    labels = np.concatenate(labels_camvid, labels_roma)

    # Shuffle dateset, then create training- and validation arrays
    img_train, img_val, label_train, label_val = train_test_split(images, labels, test_size=0.15,
                                                                  shuffle=True, random_state=helper.random_state)
    # helper.blend_images(images=img_val, labels=label_val, folder_url='datas/images_camvid/output/')

    # MODEL

    # Main model parameters
    batch_size = 16
    epochs = 1000
    input_shape = img_train.shape[1:]
    dropout_rate = 0.2
    pool_size = (2, 2)
    learning_rate = 0.001

    # Load model structure
    model = my_models.get_model_segm(input_shape, pool_size, dropout_rate)

    # TODO: custom loss!
    # Initialize model + print model
    model.compile(optimizer=Adam(lr=learning_rate), loss=categorical_crossentropy)

    my_models.train_model(model, img_train, img_val, label_train, label_val, batch_size, epochs,
                          log_path='logs/camvid', save_path='models/camvid', out_path=data_path + 'images_camvid/output/', datagen=True)


main()
