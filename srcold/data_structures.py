import os
import math
import pandas
import numpy as np
import keras

from importlib import import_module
from utils.logger import conditional_makedir
from utils.transforms import DataAugmentationTransform

# convert series to supervised learning
def series_to_supervised(data, n_length, n_stride=1):
    data = np.pad(data,((0,int((math.ceil((float(data.shape[0]) - n_length + n_stride)/n_stride) - 1)*n_stride+n_length-data.shape[0]))),'edge')
    agg  = np.zeros((math.ceil((data.shape[0] - n_length + n_stride)/n_stride),n_length))

    for i in range(0,agg.shape[0]):
        agg[i,:] = data[i*n_stride:(i*n_stride+n_length)]

    return agg


def supervised_to_series(supervised, n_length, n_stride=1):
    dtype   = supervised.dtype
    agg     = np.zeros(((supervised.shape[0]-1)*n_stride+n_length,))
    contrib = np.zeros(((supervised.shape[0]-1)*n_stride+n_length,))

    for i in range(0,supervised.shape[0]):
        agg[i*n_stride:(i*n_stride+n_length)]       += supervised[i,:]
        contrib[i*n_stride:(i*n_stride+n_length)]   += 1

    return (agg/contrib).astype(dtype)


# Data loader to un-clutter code    
def load_data(filepath):
    dic = dict()
    with open(filepath) as f:
        text = list(f)
    for line in text:
        line = line.replace(' ','').replace('\n','').replace(',,','')
        if line[-1] == ',': line = line[:-1]
        head = line.split(',')[0]
        tail = line.split(',')[1:]
        if tail == ['']:
            tail = np.asarray([])
        else:
            tail = np.asarray(tail).astype(int)

        dic[head] = tail
    return dic


# Data loader to un-clutter code    
def save_data(filepath, dic):
    with open(filepath, 'w') as f:
        for key in dic.keys():
            # f.write("%s,%s\n"%(key,dic[key].tolist()))
            f.write("{},{}\n".format(key,str(dic[key].tolist()).replace(']','').replace('[','').replace(' ','')))


def save_fiducials(fiducial, path):
    with open(path, 'w') as f:
        for key in fiducial.keys():
            f.write("%s,%s\n"%(key,str(fiducial[key].tolist()).replace('[','').replace(']','')))


# class ExecInfo(object):
#     def __init__(self, config, fold, train, valid, test):
#         self.sets           = SetInfo(train, valid, test)
#         self.paths          = PathInfo(config, fold)

#     def __repr__(self):
#         return self.__str__()


# class SetInfo(object):
#     def __init__(self, train, valid, test):
#         self.train          = train
#         self.valid          = valid
#         self.test           = test

#     def __repr__(self):
#         return self.__str__()


# class PathInfo(object):
#     def __init__(self, config, fold):
#         self.summary         = os.path.join(config.output_dir, 'Summary.txt')

#         if (config.splitting == "cross_validation") and fold != None:
#             self.logger      = os.path.join(config.output_dir, config.splitting, 'Training', 'Fold_' + str(fold + 1), 'Model.log')
#             self.state       = os.path.join(config.output_dir, config.splitting, 'Training', 'Fold_' + str(fold + 1), 'Model.hdf5')
#             self.results     = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Fold_' + str(fold + 1), 'Results_ALL.csv')
#             self.results_CV2 = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Fold_' + str(fold + 1), 'Results_ALL_CV2.csv')

#         else: # Case "all"
#             self.logger      = os.path.join(config.output_dir, config.splitting, 'Training', 'Total.log')
#             self.state       = os.path.join(config.output_dir, config.splitting, 'Training', 'Total.hdf5')
#             self.results     = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Total_ALL.csv')
#             self.results_CV2 = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Total_ALL_CV2.csv')

#         conditional_makedir(self.summary)
#         conditional_makedir(self.logger)
#         conditional_makedir(self.state)
#         conditional_makedir(self.results)
#         conditional_makedir(self.results_CV2)

#     def __repr__(self):
#         return self.__str__()


# class FoldKeys():
#     def __init__(self, train_keys=None, valid_keys=None, test_keys=None):
#         """Use fold = None for whole dataset test"""

#         self.train          = train_keys
#         self.valid          = valid_keys
#         self.test           = test_keys

#     def __str__(self):
#         s = ''
#         for k in self.__dict__.keys():
#             s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
#         return s

#     def __repr__(self):
#         return self.__str__()


# class FoldPaths():
#     def __init__(self, config, fold=None):
#         """Use fold = None for whole dataset test """

#         self.summary         = os.path.join(config.output_dir, 'Summary.txt')

#         if (config.splitting == "cross_validation") and fold != None:
#             self.logger      = os.path.join(config.output_dir, config.splitting, 'Training', 'Fold_' + str(fold + 1), 'Model.log')
#             self.state       = os.path.join(config.output_dir, config.splitting, 'Training', 'Fold_' + str(fold + 1), 'Model.hdf5')
#             self.results     = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Fold_' + str(fold + 1), 'Results_ALL.csv')
#             self.results_CV2 = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Fold_' + str(fold + 1), 'Results_ALL_CV2.csv')

#         else: # Case "all"
#             self.logger      = os.path.join(config.output_dir, config.splitting, 'Training', 'Total.log')
#             self.state       = os.path.join(config.output_dir, config.splitting, 'Training', 'Total.hdf5')
#             self.results     = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Total_ALL.csv')
#             self.results_CV2 = os.path.join(config.output_dir, config.splitting, 'Results', config.data_set, 'Total_ALL_CV2.csv')

#         conditional_makedir(self.summary)
#         conditional_makedir(self.logger)
#         conditional_makedir(self.state)
#         conditional_makedir(self.results)
#         conditional_makedir(self.results_CV2)


#     def __str__(self):
#         s = ''
#         for k in self.__dict__.keys():
#             s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
#         return s

#     def __repr__(self):
#         return self.__str__()


class WaveInformation():
    def __init__(self):
        self.wave   = pandas.DataFrame()
        self.onset  = dict()
        self.peak   = dict()
        self.offset = dict()

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()

class ExecutionInformation():
    def __init__(self, config, fold, train_keys, test_keys, evaluate):
        """Use fold = None for whole dataset test """

        # Specify fold
        self.fold                   = fold
        self.evaluate               = evaluate
        self.output_dir             = config.output_dir
        self.backend                = config.backend.lower()
        self.data_set               = config.data_set.lower()
        self.splitting              = config.splitting.lower()


        # Define train/test sets
        if self.evaluate == True: # If only evaluating, no need for validation set
            valid_keys          = []
        else:
            train_keys          = np.random.permutation(train_keys)
            valid_keys          = train_keys[:math.ceil(config.val_split*len(train_keys))]
            train_keys          = train_keys[math.ceil(config.val_split*len(train_keys)):]

        # Set paths
        self.summary        = os.path.join(self.output_dir, 'Summary.txt')

        if (self.splitting == "cross_validation") and self.fold != None:
            self.logger  = os.path.join(self.output_dir, self.backend, self.splitting, 'Training', 'Fold_' + str(self.fold + 1), 'Model.log')
            self.state   = os.path.join(self.output_dir, self.backend, self.splitting, 'Training', 'Fold_' + str(self.fold + 1), 'Model.hdf5')
            self.results = os.path.join(self.output_dir, self.backend, self.splitting, 'Results', self.data_set, 'Fold_' + str(self.fold + 1), 'Results_ALL.csv')

        else: # Case "all"
            self.logger  = os.path.join(self.output_dir, self.backend, self.splitting, 'Training', 'Total.log')
            self.state   = os.path.join(self.output_dir, self.backend, self.splitting, 'Training', 'Total.hdf5')
            self.results = os.path.join(self.output_dir, self.backend, self.splitting, 'Results', self.data_set, 'Total_ALL.csv')

        conditional_makedir(self.summary)
        conditional_makedir(self.logger)
        conditional_makedir(self.state)
        conditional_makedir(self.results)

        self.train       = train_keys
        self.valid       = valid_keys
        self.test        = test_keys

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()


class DataStorage():
    def __init__(self, dataset, validity=None):
        self.dataset    = dataset
        self.validity   = validity
        self.P          = WaveInformation()
        self.QRS        = WaveInformation()
        self.T          = WaveInformation()
        self.keys       = []

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()
    
    def init_P(self, wave, onset, peak, offset):
        self.P.wave     = wave
        self.P.onset    = onset
        self.P.peak     = peak
        self.P.offset   = offset
        self.keys       = list(set(self.keys + wave.keys().tolist()))

    def init_QRS(self, wave, onset, peak, offset):
        self.QRS.wave   = wave
        self.QRS.onset  = onset
        self.QRS.peak   = peak
        self.QRS.offset = offset
        self.keys       = list(set(self.keys + wave.keys().tolist()))

    def init_T(self, wave, onset, peak, offset):
        self.T.wave     = wave
        self.T.onset    = onset
        self.T.peak     = peak
        self.T.offset   = offset
        self.keys       = list(set(self.keys + wave.keys().tolist()))


class WaveMetricsStorage():
    def __init__(self):
        self.truepositive  = dict()
        self.falsepositive = dict()
        self.falsenegative = dict()
        self.onseterror    = dict()
        self.offseterror   = dict()
        self.dice          = dict()
        self.keys          = []

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, k):
        return self.truepositive[k], self.falsepositive[k], self.falsenegative[k], self.onseterror[k], self.offseterror[k]


class Database(object):
    def __init__(self):
        self.registries = dict()

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()    


class Registry(object):
    def __init__(self, lead_0, lead_1):
        self.lead_0 = Signal()
        self.lead_1 = Signal()

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()    


class Signal(object):
    def __init__(self):
        self.P   = Wave()
        self.QRS = Wave()
        self.T   = Wave()


class Wave(object):
    def __init__(self):
        self.signal     = []
        self.validity   = dict()
        self.onset      = []
        self.peak       = []
        self.offset     = []
        self.metrics    = WaveMetricsStorage()

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()



class MetricsStorage():
    def __init__(self):
        self.P       = WaveMetricsStorage()
        self.QRS     = WaveMetricsStorage()
        self.T       = WaveMetricsStorage()

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()

    def init_P(self, truepositive, falsepositive, falsenegative, onseterror, offseterror):
        self.P.truepositive    = truepositive
        self.P.falsepositive   = falsepositive
        self.P.falsenegative   = falsenegative
        self.P.onseterror      = onseterror
        self.P.offseterror     = offseterror

    def init_QRS(self, truepositive, falsepositive, falsenegative, onseterror, offseterror):
        self.QRS.truepositive  = truepositive
        self.QRS.falsepositive = falsepositive
        self.QRS.falsenegative = falsenegative
        self.QRS.onseterror    = onseterror
        self.QRS.offseterror   = offseterror

    def init_T(self, truepositive, falsepositive, falsenegative, onseterror, offseterror):
        self.T.truepositive    = truepositive
        self.T.falsepositive   = falsepositive
        self.T.falsenegative   = falsenegative
        self.T.onseterror      = onseterror
        self.T.offseterror     = offseterror


class ConfigParser():
    def __init__(self, config_dict, ex_id, data_path, data_set, splitting, load_weights, backend, output_dir, evaluate):
        assert backend.lower() in ('keras', 'torch', 'pytorch')

        # Retrieve information from configuration dictionary
        self.depth           = int(config_dict['A_Depth'])
        self.m_repetitions   = int(config_dict['A_Repetitions'])
        self.start_ch        = int(config_dict['A_InitChannels'])
        self.data_aug        = bool(config_dict['T_DataAug'])
        self.ms_upsampling   = bool(config_dict['A_MSUpsampling'])
        self.atrous_conv     = bool(config_dict['A_ASPP'])
        self.hyperdense      = bool(config_dict['A_HyperDense'])
        self.m_name          = str(config_dict['A_Module'])
        self.loss            = str(config_dict['T_Loss'])
        self.optimizer       = str(config_dict['T_Optimizer'])
        self.inc_rate        = float(config_dict['A_LvlGrowth'])
        self.learning_rate   = float(config_dict['T_LearningRate'])
        self.kernel_init     = str(config_dict['A_KernelInitializer'])
        self.kernel_size     = int(config_dict['A_KernelSize'])
        self.dropout_rate    = float(config_dict['A_DropoutRate'])
        self.batch_size      = int(config_dict['T_BatchSize'])
        self.out_ch          = int(config_dict['A_OutChannels'])
        self.element_size    = int(config_dict['P_ElementSize'])
        self.max_size        = int(config_dict['D_MaxSize'])
        self.maxpool         = int(config_dict['A_MaxPooling'])
        self.sampling_freq   = float(config_dict['D_fs'])
        self.stride          = int(config_dict['T_Stride'])
        self.window          = int(config_dict['T_Window'])
        self.n_epochs        = int(config_dict['T_Epochs'])
        self.lr_patience     = int(config_dict['T_LRPatience'])
        self.patience        = int(config_dict['T_Patience'])
        self.val_split       = float(config_dict['T_ValidationSplit'])
        self.seed            = int(config_dict['T_Seed'])
        self.strategy        = str(config_dict['A_Strategy'])

        if self.strategy in ('single', 'single_lead', 'single-lead', 'single lead'):
            self.in_ch       = 1
        elif self.strategy in ('multi', 'multilead', 'multi_lead', 'multi-lead', 'multi lead'):
            self.in_ch       = 2
        else:
            raise ValueError("Training strategy '" + str(self.strategy) + "' not implemented")

        # Inputs
        self.ex_id           = int(ex_id)
        self.backend         = backend
        self.data_set        = data_set
        self.data_path       = os.path.join(data_path, data_set)
        self.splitting       = splitting
        self.load_weights    = load_weights
        self.evaluate        = evaluate
        self.device          = 'cuda'
        self.output_dir      = output_dir

        if self.output_dir == None: 
            self.output_dir  = os.path.join(os.path.abspath('./Logs/'), 'Config' + str(self.ex_id))
        else:
            self.output_dir  = os.path.join(self.output_dir, 'Config' + str(self.ex_id))


    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s

    def __repr__(self):
        return self.__str__()


class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''

    def __init__(self, labels, config, data, shuffle=True):
        '''Initialization'''
        # awgn=20, spikes=(30,200), powerline=(20,50.), baseline=(-5,0.15), pacemaker=15, sat_threshold=5)
        
        # Easier indexing
        leads                   = np.concatenate((pandas.Index(labels) + '_0', pandas.Index(labels) + '_1'))

        # Set properties
        self.labels             = labels
        self.shuffle            = shuffle
        self.window             = config.window
        self.channels           = config.in_ch
        self.stride             = config.stride
        self.batch_size         = config.batch_size
        self.dataset            = data.dataset[leads]  # Much simpler index generation
        self.validity           = data.validity[leads]
        self.mask_p             = data.P.wave[leads]   # Much simpler index generation
        self.mask_qrs           = data.QRS.wave[leads] # Much simpler index generation
        self.mask_t             = data.T.wave[leads]   # Much simpler index generation

        if config.data_aug: # TO-DO: MORE PARAMETERS
            self.awgn           = 20
            self.spikes         = (25,150)
            self.powerline      = (20,50.)
            self.baseline       = (-5,0.15)
            self.pacemaker      = 20
            self.sat_threshold  = 50
        
            # Types of data augmentation considered
            self.DataAug        = [0,self.awgn,self.spikes,self.powerline,self.baseline,self.pacemaker,self.sat_threshold]
            self.DataAugLabel   = ['NoAugment','awgn','spikes','powerline','baseline','pacemaker','sat_threshold']
            self.DataAugLength  = np.asarray([x is not None for x in self.DataAug]).sum()

            # Obtain only the indices that we want
            self.DataAugLabel   = [self.DataAugLabel[i] for i in np.where(np.asarray([x is not None for x in self.DataAug]) == True)[0]]
            self.DataAug        = [self.DataAug[i] for i in np.where(np.asarray([x is not None for x in self.DataAug]) == True)[0]]
        else:
            self.DataAugLabel   = ['NoAugment']
            self.DataAugLength  = 1

        # Index-Window correspondence
        self.correspondence = dict()
        counter = 0 # Keep track of how many windows have already been stored

        # Single-lead strategy
        if self.channels == 1:      
            for key in leads:
                for w in range(len(self.validity[key][0])):
                    window_elements = max(((self.validity[key]['off'][w]-self.validity[key]['on'][w]+1)-self.window+self.stride)//self.stride,0)
                    
                    for i in range(window_elements):
                        self.correspondence[counter+i] = [key, self.validity[key]['on'][w]+self.stride*i]

                    counter += window_elements

        elif self.channels == 2:    # Two-lead strategy. Both leads will have the same validity <- selecting validity of lead '_0'
            for key in self.labels:
                for w in range(len(self.validity[key + '_0'][0])):
                    window_elements = max(((self.validity[key + '_0']['off'][w]-self.validity[key + '_0']['on'][w]+1)-self.window+self.stride)//self.stride,0)
                    
                    for i in range(window_elements):
                        self.correspondence[counter+i] = [key, self.validity[key + '_0']['on'][w]+self.stride*i]

                    counter += window_elements
        else:                       # Not implemented
            raise NotImplementedError("No database available for " + str(self.channels) + " channels")

        # Custom indexing
        self.length_no_aug  = len(self.correspondence.keys())
        self.indexes        = np.arange(len(self.correspondence.keys())*self.DataAugLength)
            
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return math.ceil(float(len(self.indexes))/self.batch_size)

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate self.indexes of the batch
        ix = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        if self.channels == 1:
            X, y = self.__batch_generation_single(ix)
        elif self.channels == 2:
            X, y = self.__batch_generation_multi(ix)
            # X = X[:,:,:,np.newaxis]
            # y = y[:,:,np.newaxis,:]
        else:
            raise NotImplementedError("No batch generation strategy for " + str(self.channels) + " channels has been devised")

        return X, y

    def on_epoch_end(self):
        '''Updates self.indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __batch_generation_single(self, ix):
        '''Generates one datapoint''' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(ix), self.window, self.channels), dtype='float32')
        y = np.empty((len(ix), self.window, 3), dtype='float32')

        # Generate data
        for i, ID in enumerate(ix):
            # Desambiguate data augmentation
            j         = ID//self.length_no_aug
            ID        = ID%self.length_no_aug
            w_info    = self.correspondence[ID]
            key       = w_info[0]
            onset     = w_info[1]

            # Store sample
            X[i,:,0]  = self.dataset[key][onset:onset+self.window]

            # Store mask
            y[i,:,0]  = self.mask_p[key].values[onset:onset+self.window]
            y[i,:,1]  = self.mask_qrs[key].values[onset:onset+self.window]
            y[i,:,2]  = self.mask_t[key].values[onset:onset+self.window]

            # Apply data augmentation:
            if not(self.DataAugLabel[j] == 'NoAugment'):
                Noise = DataAugmentationTransform(X[i,:,0], self.DataAugLabel[j], self.DataAug[j], y[i,:,1])
                X[i,:,0] += Noise

        return X, y

    def __batch_generation_multi(self, ix):
        '''Generates one datapoint''' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(ix), self.window, self.channels), dtype='float32')
        y = np.empty((len(ix), self.window, 3), dtype='float32')

        # Generate data
        for i, ID in enumerate(ix):
            # Desambiguate data augmentation
            j         = ID//self.length_no_aug
            ID        = ID%self.length_no_aug
            w_info    = self.correspondence[ID]
            key       = w_info[0]
            onset     = w_info[1]

            # Store sample
            X[i,:,1]  = self.dataset[key + '_0'][onset:onset+self.window]
            X[i,:,0]  = self.dataset[key + '_1'][onset:onset+self.window]

            # Store mask
            y[i,:,0]  = self.mask_p[key + '_0'].values[onset:onset+self.window]
            y[i,:,1]  = self.mask_qrs[key + '_0'].values[onset:onset+self.window]
            y[i,:,2]  = self.mask_t[key + '_0'].values[onset:onset+self.window]

            # Apply data augmentation:
            if not(self.DataAugLabel[j] == 'NoAugment'):
                Noise_0     = DataAugmentationTransform(X[i,:,0], self.DataAugLabel[j], self.DataAug[j], y[i,:,1])
                X[i,:,0]    += Noise_0.squeeze()
                Noise_1     = DataAugmentationTransform(X[i,:,0], self.DataAugLabel[j], self.DataAug[j], y[i,:,1])
                X[i,:,1]    += Noise_1.squeeze()

        return X, y

    def __str__(self):
        s = ''
        for k in self.__dict__.keys():
            s += '    ├> ' + str(k) + ':\t' + str(self.__dict__[k]) + '\n'
        
        return s
    
    def __repr__(self):
        return str(self)
