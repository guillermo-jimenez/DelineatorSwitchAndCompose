from typing import Callable, Tuple, List, Iterable, Union
import warnings
import torch
import torch.utils
import numpy as np
import math
import random
import pathlib
import fleetfmt
import scipy
import scipy.stats
from scipy.stats import norm
from scipy.interpolate import interp1d

import sak.signal
import sak.data
import sak.data.augmentation

def sigmoid(x: float or Iterable) -> float or np.ndarray:
    return 1/(1 + np.exp(-x))


class DatasetUnsupervised(torch.utils.data.Dataset):
    '''Unsupervised dataset. BEWARE, MUST BE USED WITH "num_workers = 0"'''

    def __init__(self, file: str, window: int, N: int = None, dtype='float32', per_db: bool = True, scale: Callable = lambda x: x):
        '''Initialization. BEWARE, MUST BE USED WITH "num_workers = 0"'''
        # Store inputs
        self.file              = pathlib.Path(file)
        self.fhandle           = self.file.open('rb')
        self.reader            = fleetfmt.FileReader(self.fhandle)
        self.keys              = list(self.reader.keys())
        # self.keymap            = reader._keymap
        # self.schema            = reader._schema
        self.window            = window
        self.dtype             = dtype
        self.N                 = N
        self.per_db            = per_db
        self.scale             = scale
        if isinstance(self.scale,str):
            self.scale         = sak.class_selector(self.scale)
            
        if self.per_db:
            self.databases         = []
            self.key_per_database  = {}
            for k in self.keys:
                db = k.split("/")[0]
                if db not in self.key_per_database:
                    self.databases.append(db)
                    self.key_per_database[db] = []
                self.key_per_database[db].append(k)
            self.__getkey      = self.__get_key_oversampling
            self.databases     = np.random.permutation(self.databases).tolist()
        else:
            self.__getkey      = self.__get_key_all
            
    def __len__(self):
        '''Denotes the number of elements'''
        return self.N
    
    def __get_key_oversampling(self, i):
        db = self.databases[i%len(self.databases)]
        idx = random.randint(0,len(self.key_per_database[db])-1)
        return self.key_per_database[db][idx]
    
    def __get_key_all(self, i):
        idx = random.randint(0,len(self.keys)-1)
        return self.keys[idx]
    
    def __getitem__(self, i):
        # Get key
        key = self.__getkey(i)
        
        # Read fragment
        fragment = self.reader.read(key)
        
        # Generate onset randomly
        onset = random.randint(0,fragment.size-self.window)
        
        # Return fragment as dict
        return {"x": sak.data.ball_scaling(fragment[onset:onset+self.window].astype(self.dtype),metric=self.scale)[None,]}


# class DatasetUnsupervisedFILES(torch.utils.data.Dataset):
#     '''Unsupervised dataset'''

#     def __init__(self, root: str, window: int, N: int = None, dtype='float32', per_db: bool = True):
#         '''Initialization'''
#         # Store inputs
#         self.root              = root
# #         self.all_files         = ['/media/guille/DADES/DADES/ECG/ALL_UNSUPERVISED/Brugada/041445/AVF_1.pkl']
# #         self.all_files         = [self.all_files[0]]*3959209
#         self.all_files         = glob.glob(os.path.join(self.root,"**","*.pkl"),recursive=True)
#         self.window            = window
#         self.dtype             = dtype
#         self.N                 = N
#         self.per_db            = per_db
            
#         if self.per_db:
#             self.databases         = []
#             self.key_per_database  = {}
#             for file in self.all_files:
#                 root,idx = os.path.split(file)
#                 root,pat = os.path.split(root)
#                 root,db  = os.path.split(root)
#                 if db not in self.key_per_database:
#                     self.databases.append(db)
#                     self.key_per_database[db] = []
#                 self.key_per_database[db].append(file)
#             self.__getkey      = self.__get_key_oversampling
#             self.databases     = np.random.permutation(self.databases).tolist()
#         else:
#             self.__getkey      = self.__get_key_all
            
#     def __len__(self):
#         '''Denotes the number of elements'''
#         return self.N
    
#     def __get_key_oversampling(self, i):
#         db = self.databases[i%len(self.databases)]
#         return random.choice(self.key_per_database[db])
    
#     def __get_key_all(self, i):
#         return random.choice(self.key_per_database[db])
    
#     def __getitem__(self, i):
#         # Get key
#         key = self.__getkey(i)
        
#         # Read fragment
#         with open(key,"rb"):
#             fragment = pickle.load(key,protocol=5)
        
#         # Generate onset randomly
#         onset = random.randint(0,fragment.size-self.window)
        
#         # Return fragment as dict
#         return {"x": fragment[None,onset:onset+self.window].astype(self.dtype)}


class DatasetQTDB(torch.utils.data.Dataset):
    '''Generates data for PyTorch'''

    def __init__(self, x, y, window, stride, dtype='float32'):
        '''Initialization'''
        assert set(x.keys()) == set(y.keys())
        # Store inputs
        self.window = window
        self.stride = stride
        self.dtype = dtype
        self.x = x
        self.y = y
        self.keys = list(x)

        # Compute size
        self.window_distribution = np.cumsum([0] + [(x[k].size - window + stride)//stride for k in x])
        self.num_windows = self.window_distribution[-1] # Extremely small performance gain

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return self.num_windows
    
    def __get_key_window(self, i):
        """Retrieve an index's key and number of window"""
        loc = np.argmax(i < self.window_distribution)
        key = self.keys[loc-1]
        win = i-self.window_distribution[loc-1]
        
        return key,win
    
    def __getitem__(self, i):
        '''Generates one datapoint''' 
        # Retrieve window location
        key,n_window = self.__get_key_window(i)

        # Compute onsets and offsets for localization
        on  = n_window*self.stride
        off = on + self.window
        
        # Retrieve data
        x = self.x[key][:,on:off]
        y = self.y[key][:,on:off]
        
        if i == self.num_windows:
            raise StopIteration
        
        return {"x": x.astype(self.dtype), "y": y.astype(self.dtype)}


class OversampledDatasetQTDB(torch.utils.data.Dataset):
    '''Generates data for PyTorch'''

    def __init__(self, x, y, window, stride, length, dtype='float32'):
        '''Initialization'''
        assert set(x.keys()) == set(y.keys())
        # Store inputs
        self.window = window
        self.stride = stride
        self.length = length
        self.dtype = dtype
        self.x = x
        self.y = y
        self.keys = list(x)

        # Compute size
        self.window_distribution = np.cumsum([0] + [(x[k].size - window + stride)//stride for k in x])
        self.num_windows = self.window_distribution[-1] # Extremely small performance gain

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return self.length
    
    def __get_key_window(self, i):
        """Retrieve an index's key and number of window"""
        loc = np.argmax(i < self.window_distribution)
        key = self.keys[loc-1]
        win = i-self.window_distribution[loc-1]
        
        return key,win
    
    def __getitem__(self, i):
        '''Generates one datapoint''' 
        # Retrieve window location
        key,n_window = self.__get_key_window(i%self.num_windows)

        # Compute onsets and offsets for localization
        on  = n_window*self.stride
        off = on + self.window
        
        # Retrieve data
        x = self.x[key][:,on:off]
        y = self.y[key][:,on:off]
        
        if i == self.length:
            raise StopIteration
        
        return {"x": x.astype(self.dtype), "y": y.astype(self.dtype)}


class Dataset(torch.utils.data.Dataset):
    '''Generates data for PyTorch'''

    def __init__(self, P, QRS, T, PQ, ST, TP, 
                 Pdistribution, QRSdistribution, Tdistribution, 
                 PQdistribution, STdistribution, TPdistribution, 
                 length = 32768, N = 2048, proba_same_morph = 0.25,
                 proba_elevation = 0.2, proba_AV_block = 0.1,
                 proba_interpolation = 0.2, proba_merge_TP = 0.25,
                 proba_merge_PQ = 0.15, proba_merge_ST = 0.25,
                 proba_mixup = 0.25, mixup_alpha = 25, mixup_beta = 5,
                 proba_TV = 0.1, proba_AF = 0.1, proba_ectopics = 0.1, 
                 proba_flatline = 0.05, proba_tachy = 0.1, 
                 proba_sinus_arrest = 0.1, proba_U_wave = 0.1,
                 ectopic_amplitude_threshold = 0.1, apply_smoothing = True,
                 tachy_maxlen = 15, elevation_range = 0.1,
                 baseline_noise_scale = 0.025, baseline_smoothing_window = 51, 
                 joint_smoothing_window = 5, convolution_ptg = 0.25,
                 amplitude_std = 0.25, interp_std = 0.15, 
                 ectopic_QRS_size = 40, proba_no_U_wave_TV = 0.85,
                 QRS_ampl_low_thres = 0.5, QRS_ampl_high_thres = 1.25,
                 scaling_metric: Callable = sak.signal.amplitude,
                 labels_as_masks = True, return_beats: bool = False,
                 relative_amplitude = True, seed = None, dtype = 'float32'):
        #### Segments ####
        # P wave
        self.P = P
        self.Pdistribution = Pdistribution
        self.Pkeys = np.array(list(P))
        # PQ wave
        self.PQ = PQ
        self.PQdistribution = PQdistribution
        self.PQkeys = np.array(list(PQ))
        # QRS wave
        self.QRS = QRS
        self.QRSkeys = np.array(list(QRS))
        self.QRSdistribution = QRSdistribution
        self.QRSsign = {k: round(float(sak.signal.signed_maxima(self.QRS[k]))) for k in self.QRS}
        self.QRS_last_sign = {}
        for k in self.QRS:
            segment = self.QRS[k]
            crossings = sak.signal.zero_crossings(segment)
            self.QRS_last_sign[k] = np.sign(sak.signal.signed_maxima(segment[crossings[-2]:crossings[-1]]))
        # ST wave
        self.ST = ST
        self.STdistribution = STdistribution
        self.STkeys = np.array(list(ST))
        # T wave
        self.T = T
        self.Tdistribution = Tdistribution
        self.Tkeys = np.array(list(T))
        # TP wave
        self.TP = TP
        self.TPdistribution = TPdistribution
        self.TPkeys = np.array(list(TP))
        
        #### Generation hyperparams ####
        # Ints'n'stuff
        if seed is None:
            self.rng = np.random
        else:
            self.rng = np.random.RandomState(seed)
        self.dtype = dtype
        self.length = length
        self.N = N
        self.cycles = self.N//32 # Rule of thumb
        self.baseline_smoothing_window = baseline_smoothing_window
        self.baseline_noise_scale = baseline_noise_scale
        self.joint_smoothing_window = joint_smoothing_window
        self.labels_as_masks = labels_as_masks
        self.amplitude_std = amplitude_std
        self.interp_std = interp_std
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.elevation_range = elevation_range
        self.ectopic_QRS_size = ectopic_QRS_size
        self.tachy_maxlen = tachy_maxlen
        self.convolution_ptg = convolution_ptg
        self.ectopic_amplitude_threshold = ectopic_amplitude_threshold
        self.apply_smoothing = apply_smoothing
        self.return_beats = return_beats
        self.relative_amplitude = relative_amplitude
        self.scaling_metric = scaling_metric
        self.QRS_ampl_low_thres = QRS_ampl_low_thres
        self.QRS_ampl_high_thres = QRS_ampl_high_thres
        if isinstance(self.scaling_metric,str): # Map to function
            self.scaling_metric = eval(self.scaling_metric)
        # Probabilities
        self.proba_TV = proba_TV
        self.proba_AF = proba_AF
        self.proba_AV_block = proba_AV_block
        self.proba_merge_TP = proba_merge_TP
        self.proba_merge_PQ = proba_merge_PQ
        self.proba_merge_ST = proba_merge_ST
        self.proba_same_morph = proba_same_morph
        self.proba_U_wave = proba_U_wave
        self.proba_no_U_wave_TV = proba_no_U_wave_TV
        self.proba_elevation = proba_elevation
        self.proba_flatline = proba_flatline
        self.proba_sinus_arrest = proba_sinus_arrest
        self.proba_interpolation = proba_interpolation
        self.proba_mixup = proba_mixup
        self.proba_tachy = proba_tachy
        self.proba_ectopics = proba_ectopics


    def __len__(self):
        '''Denotes the number of elements in the dataset'''
        return self.length


    def __getitem__(self, i):
        '''Generates one datapoint''' 
        ##### Generate globals #####
        dict_globals = self.generate_globals()
        dict_globals['IDs'] = self.generate_IDs(dict_globals)
        dict_globals['amplitudes'] = self.generate_amplitudes(dict_globals)
        dict_globals['elevations'] = self.generate_elevations(dict_globals)
        dict_globals['IDs']['extension'] = self.generate_extensions(dict_globals)
        
        ##### Generate beats, cycle by cycle #####
        beats = []
        beat_types = []
        for index in range(self.N): # Unrealistic upper limit
            mark_break = self.generate_cycle(index, dict_globals, beats, beat_types)
            if mark_break: break

        ##### Registry-wise post-operations #####
        # Concatenate signal and mask
        signal = np.concatenate(beats)
        masks = self.generate_mask(beats, beat_types, mode_bool=self.labels_as_masks) # Still to be done right

        # Apply beat elevation/depression
        signal = self.signal_elevation(signal, beats, dict_globals)
        
        # Interpolate signal & mask
        int_len = math.ceil(signal.size*dict_globals['interp_length'])
        signal = self.interpolate(signal,int_len)
        masks = self.interpolate(masks,int_len,axis=-1,kind='next').astype(masks.dtype)

        # Apply global amplitude modulation
        signal = signal*dict_globals['global_amplitude']

        # Apply whole-signal modifications
        signal = self.signal_baseline_wander(signal)
        if dict_globals['has_AF']:              signal = self.signal_AF(signal)
        if dict_globals['has_flatline']:        signal, masks = self.signal_flatline(signal, masks, dict_globals)
        
        # Apply random onset for avoiding always starting with the same wave at the same location
        on = dict_globals['index_onset']
        signal = signal[on:on+self.N]
        masks = masks[:,on:on+self.N]

        # Smooth result
        if self.apply_smoothing: signal = self.smooth(signal, self.joint_smoothing_window)

        # 6. Return
        out_dict = {}
        out_dict['x'] = signal[None,].astype(self.dtype)
        out_dict['y'] = masks.astype(self.dtype)
        if self.return_beats:
            out_dict['beats'] = beats
            out_dict['beat_types'] = beat_types
            out_dict['dict_globals'] = dict_globals
        
        return out_dict


    def generate_globals(self):
        dict_globals = {}

        # Probabilities of waves
        dict_globals['same_morph'] = self.rng.rand() < self.proba_same_morph
        dict_globals['has_flatline'] = self.rng.rand() < self.proba_flatline
        dict_globals['has_TV'] = self.rng.rand() < self.proba_TV
        dict_globals['has_AF'] = self.rng.rand() < self.proba_AF
        dict_globals['has_AV_block'] = self.rng.rand() < self.proba_AV_block
        dict_globals['has_tachy'] = self.rng.rand() < self.proba_tachy
        dict_globals['has_elevations'] = self.rng.rand() < self.proba_elevation
        dict_globals['has_sinus_arrest'] = self.rng.rand() < self.proba_sinus_arrest
        if dict_globals['has_sinus_arrest']:
            dict_globals['arrest_location'] = self.rng.randint(2,6)
            dict_globals['arrest_duration'] = self.rng.randint(3)
        
        # Logical if has TV
        if dict_globals['has_TV']: dict_globals['has_tachy'] = True
        if dict_globals['has_TV']: dict_globals['same_morph'] = True
       
        # Set hyperparameters
        dict_globals['index_onset'] = self.rng.randint(50)
        dict_globals['begining_wave'] = self.rng.randint(7)
        dict_globals['global_amplitude'] = 1.+(self.rng.randn()*self.amplitude_std)
        dict_globals['interp_length'] = np.clip(1.+(self.rng.randn()*self.interp_std),0.75,1.25)
        if dict_globals['has_flatline']: dict_globals['flatline_length'] = self.rng.randint(1,self.N//self.rng.randint(2,5))
        if dict_globals['has_flatline']: dict_globals['flatline_left'] = self.rng.randint(dict_globals['flatline_length'])
        if dict_globals['has_flatline']: dict_globals['flatline_right'] = dict_globals['flatline_length']-dict_globals['flatline_left']
        dict_globals['N'] = math.ceil((self.N+dict_globals['index_onset']-dict_globals.get('flatline_length',0))/dict_globals['interp_length'])

        return dict_globals


    def generate_IDs(self, dict_globals: dict):
        IDs = {}

        ##### Generate conditions #####
        # Generate ectopics
        if dict_globals['has_TV']: IDs['ectopics'] = np.ones((self.cycles,),dtype=bool)
        else:                      IDs['ectopics'] = self.rng.rand(self.cycles) < self.proba_ectopics

        # Merge T+P, P+QRS, QRS+T
        IDs['merge_TP'] = self.rng.rand(self.cycles) < self.proba_merge_TP*(1+dict_globals['has_tachy'])
        IDs['merge_PQ'] = self.rng.rand(self.cycles) < self.proba_merge_PQ*(1+dict_globals['has_tachy'])
        IDs['merge_ST'] = self.rng.rand(self.cycles) < self.proba_merge_ST*(1+dict_globals['has_tachy'])
        
        ##### Generate identifiers #####
        if dict_globals['same_morph']:
            IDs[  'P'] = np.array([self.rng.randint(len(  self.P))]*self.cycles, dtype=int)
            IDs[ 'PQ'] = np.array([self.rng.randint(len( self.PQ))]*self.cycles, dtype=int)
            IDs['QRS'] = np.array([self.rng.randint(len(self.QRS))]*self.cycles, dtype=int)
            IDs[ 'ST'] = np.array([self.rng.randint(len( self.ST))]*self.cycles, dtype=int)
            IDs[  'T'] = np.array([self.rng.randint(len(  self.T))]*self.cycles, dtype=int)
            IDs[ 'TP'] = np.array([self.rng.randint(len( self.TP))]*self.cycles, dtype=int)

            # If same morph, use different morphologies for the ectopic beats (for QRS and T waves)
            IDs['QRS'][IDs['ectopics']] = np.array([self.rng.randint(len(self.QRS))]*IDs['ectopics'].sum())
            IDs[  'T'][IDs['ectopics']] = np.array([self.rng.randint(len(  self.T))]*IDs['ectopics'].sum())
        else:
            IDs[  'P'] = self.rng.randint(len(  self.P), size=self.cycles)
            IDs[ 'PQ'] = self.rng.randint(len( self.PQ), size=self.cycles)
            IDs['QRS'] = self.rng.randint(len(self.QRS), size=self.cycles)
            IDs[ 'ST'] = self.rng.randint(len( self.ST), size=self.cycles)
            IDs[  'T'] = self.rng.randint(len(  self.T), size=self.cycles)
            IDs[ 'TP'] = self.rng.randint(len( self.TP), size=self.cycles)

        original_P = np.copy(IDs['P'])

        ##### Condition-specific modifiers #####
        # ~~~~~~~~~~~~~~~ ATRIAL FIBRILLATION ~~~~~~~~~~~~~~~~
        if dict_globals['has_AF'] and not dict_globals['same_morph']: 
            IDs['P'] = np.repeat(self.rng.randint(len(self.P)),self.cycles)

        # ~~~~~~~~~~~~~~~~~~~ SINUS ARREST ~~~~~~~~~~~~~~~~~~~
        filt_arrest = np.zeros((self.cycles,),dtype=bool)
        if dict_globals['has_sinus_arrest']:
            filt_arrest[dict_globals['arrest_location']:dict_globals['arrest_location']+dict_globals['arrest_duration']] = True

        # ~~~~~~~~~~~~~ VENTRICULAR TACHYCARDIA ~~~~~~~~~~~~~~
        if dict_globals['has_tachy']:
            base_len = self.rng.randint(3,self.tachy_maxlen)
            dict_globals['tachy_len'] = (self.rng.randint(-3,3,size=self.cycles)+base_len).clip(min=0)

            filt_long_TP = self.rng.rand(self.cycles) < IDs['ectopics']*0.90
            if (not dict_globals['has_TV']):
                dict_globals['tachy_len'][filt_long_TP] += self.rng.randint(60,100,size=filt_long_TP.sum())

        # ~~~~~~~~~~~~~ ATRIOVENTRICULAR BLOCK  ~~~~~~~~~~~~~~
        if (dict_globals['has_AV_block'] and dict_globals['same_morph'] and not dict_globals['has_TV'] and not dict_globals['has_AF']):
            num_skipped = self.rng.randint(1,4)
            num_normal = self.rng.randint(2,6)
            first = self.rng.randint(0,4)

            # Define which cycles have AV block
            IDs['AV_block'] = np.array([False]*first + ([True]*num_skipped+[False]*num_normal)*math.ceil(self.cycles/(num_skipped+num_normal)))
            IDs['AV_block'] = IDs['AV_block'][:self.cycles]

            # Define AV block size
            sizes = sum([self.get_segment(type,IDs[type][0]).size for type in ['P','PQ','QRS','ST','T','TP']])
            IDs['AV_sizes'] = (sizes*((1-0.05)+0.1*self.rng.rand(self.cycles))).astype(int)

            # Refine stuff
            IDs['merge_PQ'][IDs['AV_block']] = False
            IDs['merge_ST'][IDs['AV_block']] = False
            IDs['merge_TP'][IDs['AV_block']] = False
        else:
            IDs['AV_block'] = np.zeros((self.cycles,),dtype=bool)

        ##### Exceptions according to different conditions #####
        # Modify identifiers
        IDs[  'P'][filt_arrest | dict_globals['has_TV']] = -1
        IDs[ 'PQ'][filt_arrest | dict_globals['has_TV']] = -1
        IDs['QRS'][filt_arrest                         ] = -1
        IDs[ 'ST'][filt_arrest                         ] = -1
        IDs[  'T'][filt_arrest                         ] = -1
        
        # ~~~~~~~~~~~~~~~~~~~~~~ SIZES ~~~~~~~~~~~~~~~~~~~~~~~
        IDs['QRS_sizes'] = np.array([self.QRS[self.QRSkeys[id]].size for id in IDs['QRS']])
        IDs['P_sizes'] = np.array([self.P[self.Pkeys[id]].size for id in IDs['P']])
        IDs['T_sizes'] = np.array([self.T[self.Tkeys[id]].size for id in IDs['T']])


        # ~~~~~~~~~~~~~~~~~~~~~ ECTOPICS ~~~~~~~~~~~~~~~~~~~~~
        # Filter out P waves of too wide segments
        IDs['ectopics'][IDs['QRS_sizes'] >= self.ectopic_QRS_size] = True # Rule of thumb
        IDs['PQ'][IDs['ectopics']] = -1
        IDs[ 'P'][IDs['ectopics']] = -1

        # ~~~~~~~~~~~~~~~~~~ MERGE SEGMENTS ~~~~~~~~~~~~~~~~~~
        # Modify mergers
        IDs['merge_PQ'][  (IDs['P'] == -1) | (IDs['QRS'] == -1)] = False
        IDs['merge_ST'][(IDs['QRS'] == -1) |   (IDs['T'] == -1)] = False
        IDs['merge_TP'][  (IDs['P'] == -1) |   (IDs['T'] == -1)] = False
        IDs['merge_ST'][self.rng.rand(self.cycles) < IDs['ectopics']*0.25] = False

        # Re-filter out some waves
        IDs['PQ'][IDs['merge_PQ']] = -1
        IDs['ST'][IDs['merge_ST']] = -1
        IDs['TP'][IDs['merge_TP']] = -1
            
        # ~~~~~~~~~~~~~~~~~~~~~ U WAVES ~~~~~~~~~~~~~~~~~~~~~~
        # Pre-fill with no U wave
        IDs['U'] = np.full((self.cycles,),-1,dtype=int)

        # Choose randomly U's morph
        filt_U = self.rng.rand(self.cycles) < self.proba_U_wave
        IDs['U'][filt_U] = IDs['T'][filt_U]

        # More probable U wave on ectopics
        filt_ectopics = self.rng.rand(self.cycles) <= IDs['ectopics']*0.5
        IDs['U'][filt_ectopics] = IDs['T'][filt_ectopics]

        # Deactivate on AV block
        IDs['U'][IDs['AV_block']] = -1

        # Filter out most U waves on TV (leave some for good measure)
        if dict_globals['has_TV']:
            filt_U_TV = self.rng.rand(self.cycles) <= self.proba_no_U_wave_TV
            IDs['U'][filt_U_TV] = -1

        # Generate U's sign
        if dict_globals['same_morph']: IDs['U_sign'] = np.array([self.rng.choice([-1,1])]*self.cycles, dtype=int)
        else:                          IDs['U_sign'] = self.rng.choice([-1,1], size=self.cycles)

        # Regenerate P waves
        IDs['P'][IDs['AV_block']] = original_P[IDs['AV_block']]

        return IDs


    def generate_extensions(self, dict_globals: dict):
        # Retrieve IDs
        IDs = dict_globals['IDs']

        # Generate extensions
        extensions = {
            'PQ': self.rng.rand(self.cycles)*0.2+0.75,
            'ST': self.rng.rand(self.cycles)*0.2+0.75,
            'TU': self.rng.rand(self.cycles)*0.2+0.75,
            'TP': self.rng.rand(self.cycles)*0.3+0.5,
        }

        # Disgregate by conditions:
        extensions['ST'][IDs['ectopics']] = self.rng.rand(IDs['ectopics'].sum())*0.3+0.5

        return extensions


    def generate_amplitudes(self, dict_globals: dict):
        amplitudes = {
            'P'   : self.Pdistribution.rvs(self.cycles,random_state=self.rng),
            'PQ'  : self.PQdistribution.rvs(self.cycles,random_state=self.rng)/2,
            'QRS' : self.QRSdistribution.rvs(self.cycles,random_state=self.rng),
            'ST'  : self.STdistribution.rvs(self.cycles,random_state=self.rng)/2,
            'T'   : self.Tdistribution.rvs(self.cycles,random_state=self.rng),
            'TP'  : self.TPdistribution.rvs(self.cycles,random_state=self.rng)/2,
        }

        # P wave in case small amplitudes
        filter = (amplitudes['P'] < 0.03) | (amplitudes['P'] > 0.3)
        while np.any(filter):
            # Retrieve generous sample, faster than sampling twice
            new_amplitudes = self.Pdistribution.rvs(self.cycles,random_state=self.rng)
            new_amplitudes = new_amplitudes[(new_amplitudes >= 0.03) & (new_amplitudes <= 0.3)]
            # Pad/crop the new amplitudes
            pad_len = filter.sum()-new_amplitudes.size
            if   pad_len < 0: new_amplitudes = new_amplitudes[:filter.sum()]
            elif pad_len > 0: new_amplitudes = np.pad(new_amplitudes,(0,pad_len),mode='constant',constant_values=0)
            # Input into the amplitudes vector
            amplitudes['P'][filter] = new_amplitudes
            filter = (amplitudes['P'] < 0.03) | (amplitudes['P'] > 0.3)

        # QRS in case low/high voltage
        filter = (amplitudes['QRS'] < self.QRS_ampl_low_thres) | (amplitudes['QRS'] > self.QRS_ampl_high_thres)
        while np.any(filter):
            # Retrieve generous sample, faster than sampling twice
            new_amplitudes = self.QRSdistribution.rvs(self.cycles,random_state=self.rng)
            new_amplitudes = new_amplitudes[(new_amplitudes >= self.QRS_ampl_low_thres) | (new_amplitudes <= self.QRS_ampl_high_thres)]
            # Pad/crop the new amplitudes
            pad_len = filter.sum()-new_amplitudes.size
            if   pad_len < 0: new_amplitudes = new_amplitudes[:filter.sum()]
            elif pad_len > 0: new_amplitudes = np.pad(new_amplitudes,(0,pad_len),mode='constant',constant_values=0)
            # Input into the amplitudes vector
            amplitudes['QRS'][filter] = new_amplitudes
            filter = (amplitudes['QRS'] < self.QRS_ampl_low_thres) | (amplitudes['QRS'] > self.QRS_ampl_high_thres)
        amplitudes['QRS'] = amplitudes['QRS'].clip(max=1)

        # T wave in case ectopics
        filter = (amplitudes['T'] < self.ectopic_amplitude_threshold) & dict_globals['IDs']['ectopics']
        while np.any(filter):
            # Retrieve generous sample, faster than sampling twice
            new_amplitudes = self.Tdistribution.rvs(self.cycles,random_state=self.rng)
            new_amplitudes = new_amplitudes[new_amplitudes >= self.ectopic_amplitude_threshold]
            # Pad/crop the new amplitudes
            pad_len = filter.sum()-new_amplitudes.size
            if   pad_len < 0: new_amplitudes = new_amplitudes[:filter.sum()]
            elif pad_len > 0: new_amplitudes = np.pad(new_amplitudes,(0,pad_len),mode='constant',constant_values=0)
            # Input into the amplitudes vector
            amplitudes['T'][filter] = new_amplitudes
            filter = (amplitudes['T'] < self.ectopic_amplitude_threshold) & dict_globals['IDs']['ectopics']

        # T wave in case large amplitudes
        filter = (amplitudes['T'] < 0.075) | (amplitudes['T'] > 0.5)
        while np.any(filter):
            # Retrieve generous sample, faster than sampling twice
            new_amplitudes = self.Tdistribution.rvs(self.cycles,random_state=self.rng)
            new_amplitudes = new_amplitudes[(new_amplitudes >= 0.075) & (new_amplitudes <= 0.5)]
            # Pad/crop the new amplitudes
            pad_len = filter.sum()-new_amplitudes.size
            if   pad_len < 0: new_amplitudes = new_amplitudes[:filter.sum()]
            elif pad_len > 0: new_amplitudes = np.pad(new_amplitudes,(0,pad_len),mode='constant',constant_values=0)
            # Input into the amplitudes vector
            amplitudes['T'][filter] = new_amplitudes
            filter = (amplitudes['T'] < 0.075) | (amplitudes['T'] > 0.5)

        ###### LIMIT AMPLITUDE OF P WAVE W.R.T. T WAVE #####
        amplitudes['P'] = np.min(np.vstack((amplitudes['P'],np.roll(amplitudes['T']*0.75,-1))),axis=0)

        ###### CONDITION-SPECIFIC AMPLITUDES #####
        if dict_globals['same_morph']: 
            amplitudes[  'P'] = amplitudes[  'P'][0]*(self.rng.rand(self.cycles)*0.5+0.75) # Rule of thumb
            amplitudes['QRS'] = amplitudes['QRS'][0]*(self.rng.rand(self.cycles)*0.5+0.75) # Rule of thumb
            amplitudes[  'T'] = amplitudes[  'T'][0]*(self.rng.rand(self.cycles)*0.5+0.75) # Rule of thumb

        # Apply correction factor depending on P, QRS and T widths
        amplitudes['P']   *= (1 + sigmoid( (dict_globals['IDs'][  'P_sizes'] -                    35)*0.25)) # Rule of thumb
        amplitudes['P']   *= (1 + sigmoid(-(dict_globals['IDs'][  'P_sizes'] -                    15)*0.25)) # Rule of thumb
        amplitudes['QRS'] *= (1 + sigmoid( (dict_globals['IDs']['QRS_sizes'] - self.ectopic_QRS_size)*0.25)) # Rule of thumb
        amplitudes['T']   *= (1 + sigmoid( (dict_globals['IDs'][  'T_sizes'] -                    55)*0.25)) # Rule of thumb

        # Clip amplitudes
        amplitudes['P']  = amplitudes[ 'P'].clip(min=0.03,  max=0.3)
        amplitudes['PQ'] = amplitudes['PQ'].clip(           max=0.025)
        amplitudes['ST'] = amplitudes['ST'].clip(           max=0.025)
        amplitudes['T']  = amplitudes[ 'T'].clip(min=0.075, max=0.5)
        amplitudes['TP'] = amplitudes['TP'].clip(           max=0.025)

        # Generate U wave's amplitudes
        amplitudes['U'] = amplitudes['T']*(self.rng.rand(self.cycles)*0.1+0.05)

        return amplitudes


    def generate_elevations(self, dict_globals: dict):
        # Check active waves
        active = np.vstack((dict_globals['IDs']['P'],   dict_globals['IDs']['PQ'],
                            dict_globals['IDs']['QRS'], dict_globals['IDs']['ST'],
                            dict_globals['IDs']['T'],   dict_globals['IDs']['U'],
                            dict_globals['IDs']['TP'])).T != -1

        if dict_globals['has_elevations']:
            # Generate elevation template
            elevation_template = self.elevation_range*(2*self.rng.rand(self.cycles,active.shape[1])-1)

            # Refine template's results - zero if the ID is zero
            active_rows = np.any(active,axis=-1)
            elevation_template[np.logical_not(active)] = 0

            # Give more weight to P, QRS and T waves
            elevation_template[:,[0,2,4]] *= 2

            # Define correction factor for number of non-negative amplitudes
            correction_factor = np.repeat(elevation_template.sum(-1,keepdims=True),active.shape[1],-1)
            correction_factor[active_rows] /= active[active_rows].sum(-1,keepdims=True)

            # Define % elevation per active segment
            elevation_template = elevation_template - correction_factor

            # Return elevations as a list for later usage on list of beats
            elevations = elevation_template[active]
        else:
            elevations = np.zeros((active.size))

        return elevations


    def generate_cycle(self, index: int, dict_globals: dict, beats: list = [], beat_types: list = []):
        # Init output
        total_size = np.sum([beat.size for beat in beats],dtype=int)
        qrs_amplitude = dict_globals['amplitudes']['QRS'][index]
        IDs = dict_globals['IDs'] # Declutter code

        ##### Generate all waves #####
        # Define available waves
        waves = ['P','PQ','QRS','ST','T','U','TP']

        # Output structure
        cycle = {k: None for k in waves}

        # In case the first element, crop first X waves
        if (index == 0):             waves = waves[dict_globals['begining_wave']:] 
        elif IDs['AV_block'][index]: waves = ['P', 'TP']

        # Iterate over available waves
        for i,type in enumerate(waves):
            cycle[type] = self.segment_compose(index, type, dict_globals, qrs_amplitude)

        # Reverse U wave if necessary
        if cycle['U'] is not None:
            cycle['U'] *= IDs['U_sign'][index]

        #### Merge beats #####
        if index != 0:
            if IDs['merge_PQ'][index]:       cycle[  'P'], cycle['QRS'] = self.segments_convolve(cycle[  'P'], cycle['QRS'], IDs['extension']['PQ'][index], reverse='first',sign_relation='different')
            if IDs['merge_ST'][index]:       cycle['QRS'], cycle[  'T'] = self.segments_convolve(cycle['QRS'], cycle[  'T'], IDs['extension']['ST'][index], reverse='last', sign_relation='equal')
            if IDs['U'][index] != -1:        cycle[  'T'], cycle[  'U'] = self.segments_convolve(cycle[  'T'], cycle[  'U'], IDs['extension']['TU'][index], reverse=None)
            if IDs['merge_TP'][index]:
                if IDs['U'][index] != -1:    cycle[  'U'], cycle[ 'P2'] = self.segments_convolve(cycle[  'U'], cycle[  'P'], IDs['extension']['TP'][index], reverse='last', sign_relation='different')
                if IDs['U'][index] == -1:    cycle[  'T'], cycle[ 'P2'] = self.segments_convolve(cycle[  'T'], cycle[  'P'], IDs['extension']['TP'][index], reverse='last', sign_relation='different')

        ##### Add valid cycle elements to beats #####
        for i,type in enumerate(waves):
            if cycle[type] is not None:
                # Add cycle[type] to output
                beats.append(cycle[type])
                beat_types.append(type)
                total_size += cycle[type].size

            if total_size >= dict_globals['N']: return True
        return False


    def generate_mask(self, beats: list, beat_types: list, mode_bool=True):
        beat_sizes = [beat.size for i,beat in enumerate(beats)]
        # Compute cumulative sum, append zero to start
        cumsum = np.hstack(([0],np.cumsum(beat_sizes)))

        # Compute masks
        if mode_bool:
            mask = np.zeros((3, cumsum[-1]), dtype=bool)
            for i,beat_type in enumerate(beat_types):
                if   beat_type == 'P':   mask[0,cumsum[i]:cumsum[i+1]] = True
                elif beat_type == 'QRS': mask[1,cumsum[i]:cumsum[i+1]] = True
                elif beat_type == 'T':   mask[2,cumsum[i]:cumsum[i+1]] = True
        else:
            mask = np.zeros((1, cumsum[-1]), dtype='int8')
            for i,beat_type in enumerate(beat_types):
                if   beat_type == 'P':   mask[cumsum[i]:cumsum[i+1]] = 1
                elif beat_type == 'QRS': mask[cumsum[i]:cumsum[i+1]] = 2
                elif beat_type == 'T':   mask[cumsum[i]:cumsum[i+1]] = 3
        return mask


    ################# GET METHODS #################
    def get_distribution(self, type: str):
        if type ==   'P': return self.Pdistribution
        if type ==  'PQ': return self.PQdistribution
        if type == 'QRS': return self.QRSdistribution
        if type ==  'ST': return self.STdistribution
        if type ==   'T': return self.Tdistribution
        if type ==   'U': return self.Tdistribution
        if type ==  'TP': return self.TPdistribution


    def get_keys(self, type: str):
        if type ==   'P': return self.Pkeys
        if type ==  'PQ': return self.PQkeys
        if type == 'QRS': return self.QRSkeys
        if type ==  'ST': return self.STkeys
        if type ==   'T': return self.Tkeys
        if type ==   'U': return self.Tkeys
        if type ==  'TP': return self.TPkeys


    def get_waves(self, type: str):
        if type ==   'P': return self.P
        if type ==  'PQ': return self.PQ
        if type == 'QRS': return self.QRS
        if type ==  'ST': return self.ST
        if type ==   'T': return self.T
        if type ==   'U': return self.T
        if type ==  'TP': return self.TP


    def get_segment_post_function(self, type: str):
        if type ==   'P': return self.P_post_operation
        if type ==  'PQ': return self.PQ_post_operation
        if type == 'QRS': return self.QRS_post_operation
        if type ==  'ST': return self.ST_post_operation
        if type ==   'T': return self.T_post_operation
        if type ==   'U': return self.T_post_operation
        if type ==  'TP': return self.TP_post_operation
    

    def get_segment(self, type: str, id: int = None):
        # Get wave information
        waves = self.get_waves(type)
        keys = self.get_keys(type)
        
        # Default id, in case not provided
        if id is None: id = self.rng.randint(len(keys))

        # Retrieve segment for modulation
        segment = np.copy(waves[keys[id]])

        return segment


    ################# SIGNAL-LEVEL FUNCTIONS #################
    def signal_baseline_wander(self, signal: np.ndarray):
        scale = 0.5*np.abs(self.rng.randn())*self.baseline_noise_scale
        baseline = self.random_walk(scale=scale, size=signal.size, smoothing_window=self.baseline_smoothing_window)
        signal = signal + baseline
        return signal


    def signal_flatline(self, signal: np.ndarray, masks: np.ndarray, dict_globals: dict) -> Tuple[np.ndarray, np.ndarray]:
        signal = np.pad(signal, (dict_globals['flatline_left'],dict_globals['flatline_right']), mode='edge')
        if self.labels_as_masks:
            masks = np.pad(masks, ((0,0),(dict_globals['flatline_left'],dict_globals['flatline_right'])), mode='constant', constant_values=0)
        else:
            masks = np.pad(masks, (dict_globals['flatline_left'],dict_globals['flatline_right']), mode='constant', constant_values=0)
        return signal, masks


    def signal_elevation(self, signal: np.ndarray, beats: List[np.ndarray], dict_globals: dict):
        if dict_globals['has_elevations']:
            noise = []
            on = 0
            off = 0
            total_size = 0

            for i,beat in enumerate(beats):
                off += dict_globals['elevations'][i]
                noise.append(np.linspace(on,off,beat.size))
                total_size += beat.size
                on += dict_globals['elevations'][i]
                if total_size >= dict_globals['N']: break

            noise = np.concatenate(noise)
            noise = self.smooth(noise,20)
            signal += noise[:signal.size]
        return signal


    def signal_AF(self, signal: np.ndarray):
        # select random P wave as template
        pAF = self.get_segment('P')
        N = signal.size

        # Interpolate to make wider
        pAF = self.interpolate(pAF, self.rng.randint(30,60))

        # Mirror on negative
        sign = self.rng.choice([-1,1])
        template = [sign*pAF[:-1],-sign*pAF[:-1]]

        # Repeat to reach N samples
        template = np.concatenate(template*math.ceil(N/sum([template[0].size,template[1].size])))

        # Crop to size N 
        if template.size != N:
            onset = self.rng.randint(template.size-N)
            template = template[onset:onset+N]

        # Interpolate on x axis to make less "stiff"
        x = self.rng.rand(N-1)*(self.rng.randint(5,size=N-1)**self.rng.randint(5,size=N-1))
        x = np.cumsum(self.smooth(np.hstack(([0.],x)),N//16))
        x = x/np.max(x)
        template = interp1d(np.linspace(0,1,N),template)(x)

        # Apply noise 
        template = template*self.rng.randn(template.size)
        template = self.smooth(template,pAF.size//2)

        # Sample amplitude > 0.3 (to make any noticeable difference on the signal)
        amplitude = self.Pdistribution.rvs(self.cycles,random_state=self.rng)
        while not np.any((amplitude >= 0.35) & (amplitude <= 0.7)): 
            amplitude = self.Pdistribution.rvs(self.cycles,random_state=self.rng)
        amplitude = amplitude[(amplitude >= 0.35) & (amplitude <= 0.7)][0]

        # Apply AF on signal
        signal = signal+template*amplitude
        
        return signal


    ################# SEGMENT-LEVEL FUNCTIONS #################
    def P_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if segment.size < 20:
            if self.rng.rand() < 0.25:            segment = self.interpolate(segment, self.rng.randint(15,40))
        return segment


    def PQ_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        return segment


    def QRS_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if dict_globals['IDs']['ectopics'][index]: segment = self.QRS_ectopic(segment, dict_globals, index)
        return segment


    def ST_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if dict_globals['has_TV']:                 segment = sak.signal.on_off_correction(self.random_walk(size=self.rng.randint(2,16)))
        elif dict_globals['has_tachy']:            segment = self.segment_tachy(segment, dict_globals['tachy_len'][index])
        return segment


    def T_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if dict_globals['IDs']['ectopics'][index]: segment = self.T_ectopic(segment, dict_globals, index)
        if dict_globals['has_TV']:                 segment = self.interpolate(segment,round(segment.size*(np.clip(np.random.normal()*0.05+0.6,0.5,1))))
        return segment


    def TP_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if dict_globals['IDs']['AV_block'][index]: segment = sak.signal.on_off_correction(self.smooth(self.random_walk(scale=0.05,size=dict_globals['IDs']['AV_sizes'][index]),20))
        elif dict_globals['has_tachy']:            segment = self.segment_tachy(segment, dict_globals['tachy_len'][index])
        return segment


    def segment_compose(self, index: int, type: str, dict_globals: dict, qrs_amplitude: float):
        # Retrieve segment information
        if dict_globals['IDs'][type][index] == -1:
            return None
        segment = self.get_segment(type, dict_globals['IDs'][type][index])

        # Segment post-operation
        segment = self.segment_post_operation(type, segment, dict_globals, index)

        # If empty, skip to next
        if segment is None:   return
        if segment.size == 0: return

        # Apply amplitude modulation
        segment = self.segment_amplitude(index, segment, type, qrs_amplitude, dict_globals)
        
        return segment


    def segment_post_operation(self, type: str, segment: np.ndarray, dict_globals: dict, index: int):
        post_segment_fnc = self.get_segment_post_function(type)
        
        # Apply wave-specific modifications to the segment
        segment = post_segment_fnc(segment, dict_globals, index)

        # If the segment is empty, return None
        if segment.size == 0: return

        # Apply mixup (if applicable)
        if (self.rng.rand() < self.proba_mixup) and (type in ['P','QRS','T']):
            segment2 = self.get_segment(type)
            segment2 = self.segment_post_operation(type, segment2, dict_globals, index)
            if sak.signal.signed_maxima(segment) != sak.signal.signed_maxima(segment2):
                segment2 *= -1
            segment = self.segment_mixup(segment, segment2)

        # Apply interpolation (if applicable, if segment is not empty)
        if (self.rng.rand() < self.proba_interpolation) and (segment.size > 1):
            new_size = int(segment.size*(0.85+self.rng.rand()*0.3))
            segment = self.interpolate(segment, new_size)

        return segment


    def segment_amplitude(self, index: int, segment: np.ndarray, type: str, qrs_amplitude: float, dict_globals: dict):
        if type == 'QRS':
            amplitude = qrs_amplitude
        else:
            # Draw from distribution
            amplitude = dict_globals['amplitudes'][type][index] # Apply amplitude on segment

            # If relative amplitude is in place:
            if self.relative_amplitude:
                amplitude *= qrs_amplitude

        # Apply amplitude modulation to segment
        segment *= amplitude
        
        return segment


    def segment_mixup(self, segment1: np.ndarray, segment2: np.ndarray):
        # If both segments are different in size, interpolate second segment
        if segment1.size != segment2.size: segment2 = self.interpolate(segment2, segment1.size)

        # Generate a mixup segment
        (mixup_segment,_) = sak.data.augmentation.mixup(segment1,segment2,self.mixup_alpha,self.mixup_beta,rng=self.rng)

        # Scale to be in [-1,1] range
        mixup_segment = sak.data.ball_scaling(mixup_segment,metric=self.scaling_metric)

        return mixup_segment


    def segment_elevation(self, segment: np.ndarray, amplitude: float, randomize: bool = False):
        # Randomize amplitude
        if randomize: amplitude = self.rng.rand()*amplitude

        # Copy segment
        segment = np.copy(segment)

        # Apply elevation to segment (linear so far)
        segment += np.linspace(0,amplitude,segment.size)

        return segment


    def segment_tachy(self, segment: np.ndarray, nx: int):
        new_segment = segment[:nx]
        if new_segment.size != 0:
            new_segment = sak.signal.on_off_correction(new_segment)
            if new_segment.ndim == 0:
                new_segment = new_segment[:,None]
        return new_segment


    def QRS_ectopic(self, segment: np.ndarray, dict_globals: dict, index: int):
        # Compute zero crossings of QRS wave
        crossings = sak.signal.zero_crossings(segment)

        # Compute sign of T wave
        dict_globals['sign_t'] = -np.sign(sak.signal.signed_maxima(segment[crossings[-2]:crossings[-1]]))

        # Interpolate segment, widen it
        segment = self.interpolate(segment,self.rng.randint(30,70))
        return segment


    def T_ectopic(self, segment: np.ndarray, dict_globals: dict, index: int):
        # Only apply if sign w.r.t. qrs is known
        if 'sign_t' in dict_globals:
            # Retrieve sign
            sign_t = dict_globals['sign_t']

            # Check if signs differ; if so, reverse T wave
            segment *= (-1)**(np.sign(sak.signal.signed_maxima(segment)) != sign_t)

            # Delete for future overwriting
            dict_globals.pop('sign_t') # Delete from globals for next ectopic

        return segment


    def segments_convolve(self, segment1: np.ndarray, segment2: np.ndarray, 
                                extension: float, reverse: ['first', 'last', None] = 'last', 
                                sign_relation: ['equal', 'different'] = 'equal'):
        extension = int(segment2.size*extension)
        # Create a composite shape
        if segment1.size+extension < segment2.size:
            extension = segment2.size-segment1.size
        composite = np.pad(np.copy(segment1),(0,extension), mode='edge')

        # Location of segment 1's max:
        loc_max1 = np.argmax(np.abs(composite))
        
        # Reference to locate cropping point
        reference = np.copy(composite)

        # Reverse p wave if necessary
        if reverse is not None:
            sign1 = np.sign(sak.signal.signed_maxima(segment1))
            sign2 = np.sign(sak.signal.signed_maxima(segment2))

            if   sign_relation.lower() == 'equal':     condition = sign1 == sign2
            elif sign_relation.lower() == 'different': condition = sign1 != sign2
            else: raise ValueError("Invalid sign relation: {} not in ['equal', 'different']".format(sign_relation))

            if condition: 
                if   reverse.lower() == 'first': segment1 *= -1
                elif reverse.lower() ==  'last': segment2 *= -1
                else: raise ValueError("Invalid reversal strategy: {} not in ['first', 'last', None]".format(reverse))
        
        # Add second segment to composite
        composite[-segment2.size:] += segment2
        
        # Compute cropping filter
        factor1 = reference[:segment1.size]
        factor2 = composite[:segment1.size]
        normalization_factor = sak.signal.amplitude(factor1)
        filt = np.abs(factor1-factor2)/normalization_factor > 0.5 # Rule of thumb
        # Avoid issues with too sharp waves
        filt[:loc_max1+1] = False
        # In case the above filter is all false
        filt[-1] = True
        # Locate the position where the filter meets the condition
        loc = np.argmax(filt)

        return composite[:loc], composite[loc:]


    ################# GENERAL UTILITIES #################
    def interpolate(self, y: np.ndarray, new_size: int, **kwargs):
        if y.size != new_size:
            if 'axis' in kwargs:
                size = y.shape[kwargs['axis']]
            else:
                size = y.size
            x_old = np.linspace(0,1,size)
            x_new = np.linspace(0,1,new_size)
            y = interp1d(x_old,y, **kwargs)(x_new)
        return y


    def smooth(self, x: np.ndarray, window_size: int, conv_mode: str = 'same'):
        x = np.pad(np.copy(x),(window_size,window_size),mode='edge')
        window = np.hamming(window_size)/(window_size//2)
        x = np.convolve(x, window, mode=conv_mode)
        x = x[window_size:-window_size]
        return x


    def noise(self, length: int, amplitude: float, smooth: bool = True):
        x = amplitude*(self.rng.rand(length)*2-1)
        x = sak.signal.on_off_correction(x)
        if smooth: x = self.smooth(x,5)
        return x


    def random_walk(self, scale: float = 0.15, size: int = 2048, smoothing_window: int = None, conv_mode: str = 'same'):
        noise = np.cumsum(norm.rvs(scale=scale,size=size,random_state=self.rng))
        if smoothing_window is not None:
            noise = self.smooth(noise, smoothing_window, conv_mode=conv_mode)
        return noise


    def distribution_draw(self, type: str):
        distribution = self.get_distribution(type)
        if type == 'QRS': 
            amplitude = np.inf
            while (amplitude < self.QRS_ampl_low_thres) or (amplitude > self.QRS_ampl_high_thres):
                amplitude = distribution.rvs(random_state=self.rng)
            amplitude = amplitude.clip(max=1)
        else:
            amplitude = distribution.rvs(random_state=self.rng)
        return amplitude


