import datetime

from keras.callbacks import CSVLogger, TensorBoard
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Activation
from keras.layers import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import codes.my_helper as helper
import numpy as np

__batch_norm_name = 'BatchNormalization_'
__conv_name = 'Conv2D_'
__max_pooling_name = 'MaxPooling2D_'
__dropout_name = 'Dropout_'
__up_sampling_name = 'UpSampling2D_'
__conv_transpose_name = 'Conv2DTranspose_'
__softmax_name = 'Softmax_'
__flatten_name = 'Flatten_'
__dense_name = 'Dense_'


def _get_conv_encoder_part(model, input_shape, pool_size, dropout_rate, trainable=True):
    # 1 - Convolution
    model.add(BatchNormalization(input_shape=input_shape, name=__batch_norm_name + 'e_1', trainable=trainable))
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=__conv_name + '1', trainable=trainable))
    # 2
    model.add(BatchNormalization(name=__batch_norm_name + 'e_2', trainable=trainable))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=__conv_name + '2', trainable=trainable))
    model.add(MaxPooling2D(pool_size=pool_size, name=__max_pooling_name + '1', trainable=trainable))
    # 3
    model.add(BatchNormalization(name=__batch_norm_name + 'e_3', trainable=trainable))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=__conv_name + '3', trainable=trainable))
    model.add(Dropout(dropout_rate, name=__dropout_name + 'e_1', trainable=trainable))
    # 4
    model.add(BatchNormalization(name=__batch_norm_name + 'e_4', trainable=trainable))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=__conv_name + '4', trainable=trainable))
    model.add(Dropout(dropout_rate, name=__dropout_name + 'e_2', trainable=trainable))
    # 5
    model.add(BatchNormalization(name=__batch_norm_name + 'e_5', trainable=trainable))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=__conv_name + '5', trainable=trainable))
    model.add(Dropout(dropout_rate, name=__dropout_name + 'e_3', trainable=trainable))
    model.add(MaxPooling2D(pool_size=pool_size, name=__max_pooling_name + '2', trainable=trainable))
    # 6
    model.add(BatchNormalization(name=__batch_norm_name + 'e_6', trainable=trainable))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=__conv_name + '6', trainable=trainable))
    model.add(Dropout(dropout_rate, name=__dropout_name + 'e_4', trainable=trainable))
    # 7
    model.add(BatchNormalization(name=__batch_norm_name + 'e_7', trainable=trainable))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=__conv_name + '7', trainable=trainable))
    model.add(Dropout(dropout_rate, name=__dropout_name + '5', trainable=trainable))
    model.add(MaxPooling2D(pool_size=pool_size, name=__max_pooling_name + '3', trainable=trainable))


def _get_conv_decoder_part(model, pool_size, dropout_rate, prefix):
    # 1 - Transposed Convolution
    # 3 maxpooling <--> 3 upsampling
    model.add(BatchNormalization(name=prefix + __batch_norm_name + 'd_1'))
    model.add(UpSampling2D(size=pool_size, name=prefix + __up_sampling_name + '1'))
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=prefix + __conv_transpose_name + '1'))
    model.add(Dropout(dropout_rate, name=prefix + __dropout_name + 'd_1'))
    # 2
    model.add(BatchNormalization(name=prefix + __batch_norm_name + 'd_2'))
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=prefix + __conv_transpose_name + '2'))
    model.add(Dropout(dropout_rate, name=prefix + __dropout_name + 'd_2'))
    # 3
    model.add(BatchNormalization(name=prefix + __batch_norm_name + 'd_3'))
    model.add(UpSampling2D(size=pool_size, name=prefix + __up_sampling_name + '2'))
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=prefix + __conv_transpose_name + '3'))
    model.add(Dropout(dropout_rate, name=prefix + __dropout_name + 'd_3'))
    # 4
    model.add(BatchNormalization(name=prefix + __batch_norm_name + 'd_4'))
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=prefix + __conv_transpose_name + '4'))
    model.add(Dropout(dropout_rate, name=prefix + __dropout_name + 'd_4'))
    # 5
    model.add(BatchNormalization(name=prefix + __batch_norm_name + 'd_5'))
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=prefix + __conv_transpose_name + '5'))
    model.add(Dropout(dropout_rate, name=prefix + __dropout_name + 'd_5'))
    # 6
    model.add(BatchNormalization(name=prefix + __batch_norm_name + 'd_6'))
    model.add(UpSampling2D(size=pool_size, name=prefix + __up_sampling_name + '3'))
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=prefix + __conv_transpose_name + '6'))
    # 7 - Output
    model.add(BatchNormalization(name=prefix + __batch_norm_name + 'd_7'))
    model.add(Conv2DTranspose(2, (3, 3), padding='valid', strides=(1, 1), activation='relu', name=prefix + __conv_transpose_name + '7'))
    model.add(Activation(activation='softmax', name=prefix + __softmax_name + '1'))


def _get_fconn_decoder_part(model, dropout_rate, prefix):
    # Fully connected
    model.add(Flatten(name=prefix + __flatten_name + '1'))
    # 1
    model.add(Dense(128, activation='relu', name=prefix + __dense_name + '1'))
    model.add(Dropout(dropout_rate, name=prefix + __dropout_name + 'd_1'))
    # 2
    model.add(Dense(64, activation='relu', name=prefix + __dense_name + '2'))
    # 3
    model.add(Dense(32, activation='relu', name=prefix + __dense_name + '3'))
    # output - coeffs (left , right)
    model.add(Dense(6, name=prefix + __dense_name + '4'))


def get_model_segm(input_shape, pool_size, dropout_rate, decoder_prefix='', train_enc=True):

    model = Sequential()
    _get_conv_encoder_part(model, input_shape, pool_size, dropout_rate, train_enc)
    _get_conv_decoder_part(model, pool_size, dropout_rate, decoder_prefix)
    return model


def get_model_poly(input_shape, pool_size, dropout_rate, decoder_prefix='', train_enc=True):

    model = Sequential()
    _get_conv_encoder_part(model, input_shape, pool_size, dropout_rate, train_enc)
    _get_fconn_decoder_part(model, dropout_rate, decoder_prefix)
    return model


# Training part for segmentation model
def train_model(model, img_train, img_val, label_train, label_val, batch_size, epochs, log_path, save_path, out_path, datagen=False):

    model.summary()

    # Create .csv logfile from losses + Tensorboard statistics during training
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    csv_logger = CSVLogger(log_path + '_training_' + date + '.csv')

    tensor_board_cb = TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)

    # std normalization is currently disabled <--BatchNormalization before input layer
    norm = False

    # Use data randomization or not
    if datagen:
        # Run training for 'epochs' times
        for epoch in range(0, epochs):
            print(str(epoch + 1) + '/' + str(epochs) + '. epoch' )
            # One epoch
            length = len(img_train)
            rounds = length // batch_size
            # Batch by batch
            for n in range(0, rounds):
                print(str((n + 1) * batch_size) + '/' + str(length))
                image_batch = np.copy(img_train[n * batch_size:(n + 1) * batch_size])
                label_batch = np.copy(label_train[n * batch_size:(n + 1) * batch_size])
                image_batch, label_batch = helper.transform_batch(image_batch, label_batch,
                                                                  horizontal_flip=True, rotation_range=5.0,
                                                                  channel_shift_range=0.2, samplewise_std_normalization=norm)

                model.fit(image_batch, label_batch, verbose=1,
                          validation_data=(img_val, label_val), callbacks=[csv_logger])

            if length % batch_size != 0:
                print(str(length) + '/' + str(length))
                image_batch = np.copy(img_train[rounds * batch_size:])
                label_batch = np.copy(label_train[rounds * batch_size:])
                image_batch, label_batch = helper.transform_batch(image_batch, label_batch,
                                                                  horizontal_flip=True, rotation_range=5,
                                                                  channel_shift_range=0.2, samplewise_std_normalization=norm)

                model.fit(image_batch, label_batch, verbose=1,
                          validation_data=(img_val, label_val), callbacks=[csv_logger])

            if epoch % 10 == 0:
                helper.blend_images(images=img_val[:10], model=model, folder_url=out_path, samplewise_std_normalization=norm)
    else:
        # Model training
        model.fit(img_train, label_train, batch_size=batch_size, epochs=epochs,
                  verbose=1, validation_data=(img_val, label_val), callbacks=[csv_logger, tensor_board_cb])

    # Save model with structure + model weights only
    model.save(save_path + '_full.h5')
    model.save_weights(save_path + '_weights.h5')

    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(score)

    # Create output images
    helper.blend_images(images=img_val, model=model, folder_url=out_path, samplewise_std_normalization=norm)
