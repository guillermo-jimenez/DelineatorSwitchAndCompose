import tensorflow
import keras.optimizers
import keras.initializers
from utils.losses import *

def select_loss(loss):
    switcher = {
        'dice'              : DiceLoss,
        'jaccard'           : JaccardLoss
    }
    return switcher.get(loss.lower())    


def select_optimizer(optim):
    switcher = {
        'adadelta'          : keras.optimizers.Adadelta,
        'adagrad'           : keras.optimizers.Adagrad,
        'adam'              : keras.optimizers.Adam,
        'adamax'            : keras.optimizers.Adamax,
        'rmsprop'           : keras.optimizers.RMSprop,
        'sgd'               : keras.optimizers.SGD,
        'nadam'             : keras.optimizers.Nadam
    }
    return switcher.get(optim.lower())    


def select_kernel_initializer(kernel_init):
    switcher = {
        'zeros'             : keras.initializers.Zeros(),
        'ones'              : keras.initializers.Ones(),
        'orthogonal'        : keras.initializers.Orthogonal(),
        'variance_scaling'  : keras.initializers.VarianceScaling(),
        'truncated_normal'  : keras.initializers.TruncatedNormal(),
        'random_uniform'    : keras.initializers.RandomUniform(),
        'random_normal'     : keras.initializers.RandomNormal(),
        'identity'          : keras.initializers.Identity(),
        'glorot_normal'     : keras.initializers.glorot_normal(),
        'xavier_normal'     : keras.initializers.glorot_normal(),
        'glorot_uniform'    : keras.initializers.glorot_uniform(), 
        'xavier_uniform'    : keras.initializers.glorot_uniform(),
        'he_normal'         : keras.initializers.he_normal(),
        'he_normal'         : keras.initializers.he_uniform(),
        'lecun_normal'      : keras.initializers.lecun_normal(),
        'lecun_uniform'     : keras.initializers.lecun_uniform()
    }
    return switcher.get(kernel_init.lower())

