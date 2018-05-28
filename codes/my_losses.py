import tensorflow as tf
from keras.backend.common import epsilon
import keras.backend as K


rate = 100
white_idx = 1


def weighted_binary_crossentropy(y_true, y_pred):
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= tf.reduce_sum(y_pred,
                            len(y_pred.get_shape()) - 1,
                            True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    axis = len(y_pred.get_shape()) - 1

    white_true = tf.gather(params=y_true, indices=[white_idx], axis=axis)
    black_true = tf.gather(params=y_true, indices=[1 - white_idx], axis=axis)

    white_pred = tf.gather(params=y_pred, indices=[white_idx], axis=axis)
    black_pred = tf.gather(params=y_pred, indices=[1 - white_idx], axis=axis)

    return - tf.reduce_sum((black_true * tf.log(black_pred) + white_true * tf.log(white_pred) * (rate - 1)) / rate,
                           axis=axis)


def mean_absolute_error_coeff(y_true, y_pred):
    mult = tf.constant([10**5, 10**2, 10**-2,
                        10**5, 10**1, 10**-2])
    y_true = tf.multiply(y_true, mult)
    y_pred = tf.multiply(y_pred, mult)
    return K.mean(K.abs(y_pred - y_true), axis=-1)
