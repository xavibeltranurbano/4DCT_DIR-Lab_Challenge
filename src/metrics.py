# -----------------------------------------------------------------------------
# Metrics file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K

def mean_dice_coef(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(y_pred, 4)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    diceCSF = single_dice_coef(y_true, y_pred, 1)
    diceGM = single_dice_coef(y_true, y_pred, 2)
    diceWM = single_dice_coef(y_true, y_pred, 3)
    return (diceCSF + diceGM + diceWM) / 3.0

def single_dice_coef(y_true, y_pred, nclass, smooth=1.):
    y_true_f = K.flatten(y_true[..., nclass])
    y_pred_f = K.flatten(y_pred[..., nclass])

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def csf(y_true, y_pred, smooth=1.):
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(y_pred, 4)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return single_dice_coef(y_true, y_pred, 1)

def wm(y_true, y_pred, smooth=1.):
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(y_pred, 4)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return single_dice_coef(y_true, y_pred, 3)

def gm(y_true, y_pred, smooth=1.):
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(y_pred, 4)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return single_dice_coef(y_true, y_pred, 2)

def bckgrd(y_true, y_pred, smooth=1.):
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(y_pred, 4)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return single_dice_coef(y_true, y_pred, 0)


# Weighted categorical crossentropy [1,10,3,3,] corresponding to [bckg,csf,gm,wm]
def categoricalCrossentropy(y_true, y_pred, weights=[1, 10, 3, 3]):
    # scale predictions so that the class probabilities of each sample sum to 1
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

    # clip to prevent NaN's and Inf's
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Apply weights to each class
    loss = y_true * tf.math.log(y_pred) * weights

    # Compute the weighted loss
    loss = -tf.reduce_sum(loss, axis=-1)
    return tf.reduce_mean(loss)
