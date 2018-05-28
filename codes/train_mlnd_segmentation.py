import pickle
import cv2
import numpy as np
from keras.callbacks import TensorBoard
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

import codes.my_helper as helper
import codes.my_models as my_model

# TODO: megcsinálni

def main():
    ## PREPARE INPUTS
    # list (116,) of (1024, 1280, 3) nparrays
    images = pickle.load(open('../datas/images_mlnd/image/full_CNN_train.p', "rb"))
    labels = pickle.load(open('../datas/images_mlnd/label/full_CNN_labels.p', "rb"))
    images = images[:300]
    labels = labels[:300]

    size = (320, 256)
    images = helper.resize_images(images, size)
    labels = helper.resize_images(labels, size)

    images = np.array(images, dtype=np.uint8) / 255

    labels = np.array(labels, dtype=np.uint8)
    labels = np.stack((labels, ) * 3, -1)

    labels_path = "../datas/labels_transformed_mlnd_fcn.p"
    if os.path.exists(labels_path):
        labels = pickle.load(open(labels_path, "rb"))
    else:
        labels = helper.rgb_to_classes(labels)
        pickle.dump(labels, open(labels_path, "wb"))

    img_train, img_val, label_train, label_val = train_test_split(images, labels, test_size=0.15,
                                                                  shuffle=True, random_state=helper.random_state)

    # PARAMETERS
    batch_size = 16
    epochs = 100
    input_shape = img_train[0].shape
    dropout_rate = 0.2
    pool_size = (2, 2)
    learning_rate = 0.001

    # TODO: prefix kivétele <-- így csak felülre tölti be a súlyokat
    model = my_model.get_model_segm(input_shape, pool_size, dropout_rate, decoder_prefix='mlnd_fcn', train_enc=True)    # train_enc=False

    # TRAIN MODEL + EVALUATION + SAVE MODEL
    tensor_board_cb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

    model.compile(optimizer=Adam(lr=learning_rate), loss=categorical_crossentropy)

    # model.load_weights('../models/roma_weights.001.500.h5', by_name=True)

    model.summary()

    model.fit(img_train, label_train, batch_size=batch_size,
              epochs=epochs, verbose=1, validation_data=(img_val, label_val), callbacks=[tensor_board_cb])

    model.save('../models/mlnd_full_fcn.h5')
    model.save_weights('../models/mlnd_weights_fcn.h5')

    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(score)

    # Save predicted images blended with original images
    helper.blend_images(images=img_val, model=model, folder_url='../datas/images_mlnd/output/')


main()