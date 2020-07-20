import numpy as np
import keras.backend as K
import tensorflow as tf


def DiceLoss(y_true, y_pred):
    smooth = np.finfo(y_true.dtype.as_numpy_dtype).eps
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1, keepdims=True)
    union = K.sum(y_true_f, axis=-1, keepdims=True) + K.sum(y_pred_f, axis=-1, keepdims=True)
    dice  = K.mean((2. * intersection + 1. + smooth) / (union + 1. + smooth))
    return 1-dice


def JaccardLoss(y_true, y_pred):
    smooth = np.finfo(y_true.dtype.as_numpy_dtype).eps
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1, keepdims=True)
    union = K.sum(y_true_f, axis=-1, keepdims=True) + K.sum(y_pred_f, axis=-1, keepdims=True)
    jacc  = K.mean((intersection + smooth) / (union - intersection + smooth))
    return 1-jacc
