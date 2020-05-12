# -*- coding: utf-8 -*-

import numpy as np
import numpy.random
import keras
import csv
import keras.layers
from utils.disambiguator import select_kernel_initializer
from utils.modules import StemModule
from utils.modules import AtrousMiddleModule
from utils.modules import LevelModule
from utils.modules import PoolingModule
from utils.modules import OutputModule

class FlatNet():
    '''Model generator'''

    def __init__(self, config):
        '''Initialization'''
        
        assert config.m_name in ('vanilla', 'residual', 'xception')

        # Parameters
        self.sig_shape      = (config.window,config.in_ch)
        self.m_name         = config.m_name
        self.m_repetitions  = config.m_repetitions
        self.ms_upsampling  = config.ms_upsampling
        self.hyperdense     = config.hyperdense
        self.atrous_conv    = config.atrous_conv
        self.maxpool        = False
        self.out_ch         = config.out_ch
        self.dropout_rate   = config.dropout_rate
        self.start_ch       = config.start_ch
        self.kernel_size    = config.kernel_size
        self.depth          = config.depth
        self.inc_rate       = config.inc_rate
        self.kernel_init    = select_kernel_initializer(config.kernel_init)

        # Declutter code -> essentially the same information in every single function call
        self.constant_info  = (self.kernel_size, self.dropout_rate, self.kernel_init)


    def __skipped_connection(self, model, level):
        # Add last element of the encoder
        vector = [model[level-1][-1]]

        if self.ms_upsampling:
            # Add all possible combinations of upsamplings from previous levels in the decoder
            vector += [keras.layers.UpSampling1D(size=2**(i+1))(model[level + i][-1]) for i in range((self.depth) - level)]
        else:
            # Upsample last layer
            vector += [keras.layers.UpSampling1D()(model[level][-1])]
        
        # Concatenate result
        return keras.layers.Concatenate()(vector)


    def __input_selector(self, x):
        if self.hyperdense and (len(x) != 1):
            return keras.layers.Concatenate()(x)
        else:
            return x[-1]


    def __level_operation(self, x, level):
        # If single tensor, add to list for posterior concatenation
        for j in range(self.m_repetitions):
            x.append(LevelModule(self.m_name)(self.__input_selector(x),
                                              int(self.inc_rate**level)*self.start_ch,
                                              *self.constant_info))
                     
        return x

    
    def create_model(self):
        # Storage of output tensors
        model = [[] for i in range(self.depth)]
        
        # Define model's input
        model[0].append(keras.layers.Input(shape=self.sig_shape))
        
        # Add stem to bridge the gap between segmentation and classification architectures
        model[0].append(StemModule(self.m_name)(model[0][-1], self.start_ch, *self.constant_info))

        ######################### ENCODER ##########################
        for level in range(self.depth):
            # Fill level with convolutional blocks
            model[level] = self.__level_operation(model[level], level)

            # Start the next level's module
            if (level+1) != self.depth:
                model[level+1].append(PoolingModule(self.m_name)(self.__input_selector(model[level]), int(self.inc_rate**(level+1))*self.start_ch, *self.constant_info))

        ######################### BOTTLENECK ##########################
        if self.atrous_conv:
            # Withdraw last operation
            model[-1].pop()

            # Include atrous convolution
            model[-1].append(AtrousMiddleModule(self.m_name)(self.__input_selector(model[level]), int(self.inc_rate**(self.depth-1))*self.start_ch, *self.constant_info))
            
        ######################### DECODER ##########################
        # Upsample last tensor of the bottleneck
        model[-2].append(self.__skipped_connection(model, self.depth-1))
        
        # Traverse the blocks in reverse order
        for level in range(self.depth-1)[::-1]:
            # Fill level with convolutional blocks
            model[level] = self.__level_operation(model[level], level)

            # Start the next level's module with (multi-scale?) upsampling and skipped connection
            if level != 0:
                model[level-1].append(self.__skipped_connection(model, level))
                
        # Apply final convolution with sigmoid activation function
        model[0].append(OutputModule(self.m_name)(self.__input_selector(model[0]), int(self.out_ch), 1, self.dropout_rate, self.kernel_init))
        
        return keras.Model(inputs=model[0][0], outputs=model[0][-1])


    def Export(self, path):
        with open(path,'wb') as f:
            printDict                                               = dict()
            printDict['Input_SignalShape:                       ']  = self.sig_shape
            printDict['Architecture_IsHyperDense:               ']  = self.hyperdense
            printDict['Architecture_Depth:                      ']  = self.depth
            printDict['Architecture_OutputChannels:             ']  = self.out_ch
            printDict['Architecture_StartingChannels:           ']  = self.start_ch
            printDict['Architecture_ChannelIncrementRate:       ']  = self.inc_rate
            printDict['Architecture_KernelInitializer:          ']  = self.kernel_init
            printDict['Module_Name:                             ']  = self.m_name
            printDict['Module_Repetitions:                      ']  = self.m_repetitions
            printDict['Module_KernelSize:                       ']  = self.kernel_size

            w = csv.writer(f)
            w.writerows(printDict.items())



