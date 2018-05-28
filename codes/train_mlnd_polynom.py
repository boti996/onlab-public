import datetime
import pickle

import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.losses import mean_absolute_error, mean_absolute_percentage_error
from keras.optimizers import Adam

import codes.my_helper as helper
import codes.my_models as my_model


def main():

    # DATA

    # TODO: több RAM kéne a 10-15% validation adatnak, MÓNI! ¯\_(:-)_/¯ --> próba: 1500kép/csomagos kép szerializációt gyártani
    # Train labels
    # Images are serialized in 1000 images/package format
    pack_size = 1000
    data_path = '../datas/'
    labels_file = open(data_path + 'images_mlnd/label/my_coeff_labels.p', "rb")
    labels = pickle.load(labels_file)
    labels = np.multiply(labels, helper.mlnd_norm)
    labels_file.close()
    # Validation labels
    label_val = labels[:pack_size]

    # Train images
    images_path = data_path + 'images_mlnd/image/my_coeff_images.p'
    images_file = open(images_path, "rb")
    # Validation images
    img_val = np.array(pickle.load(images_file)) / 255
    images_file.close()

    # PARAMETERS
    batch_size = 40
    iterations = 5
    epochs = 1
    input_shape = img_val[0].shape
    dropout_rate = 0.2
    pool_size = (2, 2)
    learning_rate = 0.001


    model = my_model.get_model_poly(input_shape, pool_size, dropout_rate, decoder_prefix='mlnd_poly_', train_enc=True)

    # TRAIN MODEL + EVALUATION + SAVE MODEL
    tensor_board_cb = TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)

    model.compile(optimizer=Adam(lr=learning_rate), loss=mean_absolute_error)   # mean_absolute_percentage_error

    # Load pretrained parameters
    model_path = '../models/roma_weights.001.500.h5'
    # model.load_weights(model_path, by_name=True)

    model.summary()

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    csv_logger = CSVLogger('../logs/roma_training_' + date + '.log')

    for iteration in range(0, iterations):
        # Read train data in packs
        images_file = open(images_path, "rb")
        # Drop first image package --> validation images
        pickle.load(images_file)

        # Don't use first label package --> validation labels
        n = 1
        # Read image- and label packages
        while True:
            print(str(iteration+1)+"/"+str(iterations)+" iteration, "+str(n)+". epoch")
            try:
                img_train = pickle.load(images_file)
                img_train = np.array(img_train, dtype=np.uint8) / 255
                label_train = labels[n * pack_size:(n + 1) * pack_size]
                n += 1

                model.fit(img_train, label_train, batch_size=batch_size,
                          epochs=epochs, verbose=1, validation_data=(img_val, label_val),
                          callbacks=[])

            except EOFError:
                break

        images_file.close()

    model.save('../models/mlnd_full_poly.h5')
    model.save_weights('../models/mlnd_weights_poly.h5')

    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(score)

    # Save predicted images blended with original images
    helper.blend_images_polynom(images=img_val, model=model, folder_url=data_path + 'images_mlnd/output/')


main()
