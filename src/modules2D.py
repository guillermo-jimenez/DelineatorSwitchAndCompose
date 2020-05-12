# Definition of several convolutional modules
# Normal convolution, Residual, ResNeXt, Inception, Amoeba...
# From [this link](https://towardsdatascience.com/history-of-convolutional-blocks-in-simple-code-96a7ddceac0c).

import numpy as np
import keras
import keras.layers
import keras.backend
import keras.backend as K
import tensorflow as tf


def soft_orthogonal_regularizer(d_rate=0.01):
    def aux_so(w):
        S, C, M = K.int_shape(w)
        wf = K.reshape(w, [S*C, M])
        o = K.dot(K.transpose(wf),wf) - K.eye(M, dtype=wf.dtype)
        n = keras.backend.square(o)
        n = keras.backend.sum(n)
        n = keras.backend.sqrt(n)
        out = d_rate*n

        return out

    return aux_so

def spectral_res_iso_reg(d_rate=0.01, w_rate=1e-4):
    def aux_srip(w):
        inp_shape = K.int_shape(w)
        row_dims = inp_shape[0]*inp_shape[1]
        col_dims = inp_shape[2]
        
        # Reshape w into square matrix
        w = K.reshape(w, (row_dims,col_dims))

        # Compute the norm of the weight matrix
        n = K.dot(K.transpose(w),w) - K.eye(col_dims, dtype=w.dtype)

        # Compute vector of random uniform values
        v = K.random_uniform((col_dims,1))

        v1 = K.dot(n, v)
        norm1 = K.sum(K.square(v1))**0.5
        v2 = tf.divide(v1,norm1)
        v3 = K.dot(n,v2)

        return d_rate*(K.sum(K.square(v3))**0.5) + w_rate*(K.sum(K.square(w))**0.5)

    return aux_srip


################################################################################################################################################################################
def StemModule(m_name):
    if m_name.lower() == 'vanilla':    ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'residual': ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'xception': ConvolutionalOperation = keras.layers.SeparableConv2D
    else: raise ValueError('Module name not correctly specified')

    def Stem(x, n_filters, ker_size, dropout_rate, kernel_init):
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='same',
                                   kernel_initializer=kernel_init)(x)

        m = keras.layers.ReLU()(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)

        return m

    return Stem



################################################################################################################################################################################
def LevelModule(m_name):
    if   m_name.lower() == 'vanilla':  ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'residual': ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'xception': ConvolutionalOperation = keras.layers.SeparableConv2D
    else: raise ValueError('Module name not correctly specified')

    def res_OutputOperation(x, m, n_filters, dropout_rate):
        if (x.shape[-1] != n_filters):
            x = keras.layers.ReLU()(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SpatialDropout2D(dropout_rate)(x)
            x = ConvolutionalOperation(n_filters, kernel_size=(1,1), padding='same')(x)

        return keras.layers.add([x, m])

    if   m_name.lower() == 'vanilla':  OutputOperation = lambda x, m, n_filters, dropout_rate: m
    elif m_name.lower() == 'residual': OutputOperation = lambda x, m, n_filters, dropout_rate: m # res_OutputOperation(x, m, n_filters, dropout_rate)
    elif m_name.lower() == 'xception': OutputOperation = lambda x, m, n_filters, dropout_rate: m # res_OutputOperation(x, m, n_filters, dropout_rate)
    else: raise ValueError('Module name not correctly specified')

    def LevelBlock(x, n_filters, ker_size, dropout_rate, kernel_init):
        m = keras.layers.ReLU()(x)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)

        m = keras.layers.ReLU()(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)

        return OutputOperation(x, m, n_filters, dropout_rate)

    return LevelBlock

     
def AtrousMiddleModule(m_name):
    if   m_name.lower() == 'vanilla':  ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'residual': ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'xception': ConvolutionalOperation = keras.layers.SeparableConv2D
    else: raise ValueError('Module name not correctly specified')

    def AtrousBlock(x, n_filters, ker_size, dropout_rate, kernel_init):
        m = keras.layers.ReLU()(x)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)

        m    = keras.layers.ReLU()(m)
        m    = keras.layers.BatchNormalization()(m)
        m    = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m1   = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   dilation_rate=(1,1), 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)
        m6   = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   dilation_rate=(6,1), 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)
        m12  = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   dilation_rate=(12,1), 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)
        m18  = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   dilation_rate=(18,1), 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)

        mgap = keras.layers.GlobalAveragePooling2D()(m1) if (m.shape[1] != n_filters) else keras.layers.GlobalAveragePooling2D()(m)
        mgap = keras.layers.RepeatVector(m.shape[1:3])(mgap)
        m    = keras.layers.Concatenate()([m1,m6,m12,m18,mgap])

        return m

    return AtrousBlock
      

################################################################################################################################################################################
def MergeModule(m_name):
    if   m_name.lower() == 'vanilla':  ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'residual': ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'xception': ConvolutionalOperation = keras.layers.SeparableConv2D
    else: raise ValueError('Module name not correctly specified')

    def MergeBlock(x, n_filters, ker_size, dropout_rate, kernel_init):
        m = keras.layers.ReLU()(x)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   (1,2), 
                                   padding='valid',
                                   kernel_initializer=kernel_init)(m)

        return m

    return MergeBlock



################################################################################################################################################################################
def PoolingModule(m_name):
    if   m_name.lower() == 'vanilla':  ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'residual': ConvolutionalOperation = keras.layers.Conv2D
    elif m_name.lower() == 'xception': ConvolutionalOperation = keras.layers.SeparableConv2D
    else: raise ValueError('Module name not correctly specified')

    def res_OutputOperation(x, m, n_filters, dropout_rate):
        x = keras.layers.ReLU()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(dropout_rate)(x)
        x = ConvolutionalOperation(n_filters, 
                                   kernel_size=(1,1), 
                                   strides=(2,1), 
                                   padding='same')(x)

        return keras.layers.add([x, m])

    if m_name.lower()   == 'vanilla':  OutputOperation = lambda x, m, n_filters, dropout_rate: m
    elif m_name.lower() == 'residual': OutputOperation = lambda x, m, n_filters, dropout_rate: m # res_OutputOperation(x, m, n_filters, dropout_rate)
    elif m_name.lower() == 'xception': OutputOperation = lambda x, m, n_filters, dropout_rate: m # res_OutputOperation(x, m, n_filters, dropout_rate)
    else: raise ValueError('Module name not correctly specified')


    def PoolingBlock(x, n_filters, ker_size, dropout_rate, kernel_init, strides=(2,1)):
        m = keras.layers.ReLU()(x)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)

        m = keras.layers.ReLU()(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='same',
                                   kernel_initializer=kernel_init)(m)

        m = keras.layers.MaxPooling2D(pool_size=ker_size, padding='same', strides=strides)(m)
        
        return OutputOperation(x, m, n_filters, dropout_rate)

    return PoolingBlock



def OutputModule(m_name):
    if m_name.lower() == 'vanilla':  
        ConvolutionalOperation = keras.layers.Conv2D
        OutputOperation        = lambda x, m: m
    elif m_name.lower() == 'residual': 
        ConvolutionalOperation = keras.layers.Conv2D
        OutputOperation        = lambda x, m: keras.layers.add([x, m])
    elif m_name.lower() == 'xception': 
        ConvolutionalOperation = keras.layers.SeparableConv2D
        OutputOperation        = lambda x, m: keras.layers.add([x, m])
    else:
        raise ValueError('Module name not correctly specified')

    def OutputBlock(x, n_filters, ker_size, dropout_rate, kernel_init):
        m = keras.layers.ReLU()(x)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.SpatialDropout2D(dropout_rate)(m)
        m = ConvolutionalOperation(n_filters, 
                                   ker_size, 
                                   padding='valid',
                                   activation='sigmoid', 
                                   kernel_initializer=kernel_init)(m)
        
        return m

    return OutputBlock

