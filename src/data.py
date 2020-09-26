from typing import Callable, Tuple
import torch
import torch.utils
import numpy as np
import math
import scipy
import scipy.stats
from scipy.stats import norm
from scipy.interpolate import interp1d

import utils.signal
import utils.data
import utils.data.augmentation

class Dataset(torch.utils.data.Dataset):
    '''Generates data for PyTorch'''

    def __init__(self, P, QRS, T, PQ, ST, TP, 
                 Pdistribution, QRSdistribution, Tdistribution, 
                 PQdistribution, STdistribution, TPdistribution, 
                 length = 32768, N = 2048, noise = 0.005, proba_no_P = 0.25,
                 proba_no_QRS = 0.01, proba_no_PQ = 0.15, 
                 proba_no_ST = 0.15, proba_same_morph = 0.2,
                 proba_elevation = 0.2, proba_interpolation = 0.2,
                 proba_mixup = 0.25, mixup_alpha = 1.0, mixup_beta = 1.0,
                 proba_TV = 0.05, proba_AF = 0.05, proba_ectopics = 0.25, 
                 ectopic_amplitude_threshold = 0.1, apply_smoothing = True,
                 proba_flatline = 0.05, proba_tachy = 0.05, 
                 tachy_maxlen = 10, proba_baseline_wander = 0.5, 
                 amplitude_std = 0.25, interp_std = 0.25,
                 smoothing_window = 51, labels_as_masks = True, 
                 QRS_ampl_low_thres = 0.1, QRS_ampl_high_thres = 1.15,
                 scaling_metric: Callable = utils.signal.amplitude,
                 return_beats: bool = False):
        #### Segments ####
        # P wave
        self.P = P
        self.Pdistribution = Pdistribution
        self.Pkeys = list(P)
        # PQ wave
        self.PQ = PQ
        self.PQdistribution = PQdistribution
        self.PQkeys = list(PQ)
        # QRS wave
        self.QRS = QRS
        self.QRSkeys = list(QRS)
        self.QRSdistribution = QRSdistribution
        # ST wave
        self.ST = ST
        self.STdistribution = STdistribution
        self.STkeys = list(ST)
        # T wave
        self.T = T
        self.Tdistribution = Tdistribution
        self.Tkeys = list(T)
        # TP wave
        self.TP = TP
        self.TPdistribution = TPdistribution
        self.TPkeys = list(TP)
        
        #### Generation hyperparams ####
        # Ints'n'stuff
        self.length = length
        self.N = N
        self.cycles = self.N//32 # Rule of thumb
        self.noise = noise
        self.proba_baseline_wander = proba_baseline_wander
        self.smoothing_window = smoothing_window
        self.labels_as_masks = labels_as_masks
        self.amplitude_std = amplitude_std
        self.interp_std = interp_std
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.tachy_maxlen = tachy_maxlen
        self.ectopic_amplitude_threshold = ectopic_amplitude_threshold
        self.apply_smoothing = apply_smoothing
        self.return_beats = return_beats
        self.scaling_metric = scaling_metric
        self.QRS_ampl_low_thres = QRS_ampl_low_thres
        self.QRS_ampl_high_thres = QRS_ampl_high_thres
        if isinstance(self.scaling_metric,str): # Map to function
            self.scaling_metric = eval(self.scaling_metric)
        # Probabilities
        self.proba_no_P = proba_no_P
        self.proba_no_QRS = proba_no_QRS
        self.proba_no_PQ = proba_no_PQ
        self.proba_no_ST = proba_no_ST
        self.proba_TV = proba_TV
        self.proba_AF = proba_AF
        self.proba_same_morph = proba_same_morph
        self.proba_elevation = proba_elevation
        self.proba_flatline = proba_flatline
        self.proba_interpolation = proba_interpolation
        self.proba_mixup = proba_mixup
        self.proba_tachy = proba_tachy
        self.proba_ectopics = proba_ectopics


    def __len__(self):
        '''Denotes the number of elements in the dataset'''
        return self.length


    def smooth(self, x: np.ndarray, window: int, conv_mode: str = 'same'):
        window = np.hamming(window)/(window//2)
        x = np.convolve(x, window, mode=conv_mode)
        return x


    def random_walk(self, scale: float = 0.01**(2*0.5), size: int = 2048, smoothing_window: int = None, conv_mode: str = 'same'):
        noise = np.cumsum(norm.rvs(scale=scale,size=size))
        if smoothing_window is not None:
            window = np.hamming(smoothing_window)/(smoothing_window//2)
            noise = np.convolve(noise, window, mode=conv_mode)
        return noise


    def trail_onset(self, segment: np.ndarray, onset: float):
        onset = onset-segment[0]
        off = onset-segment[0]+segment[-1]
        segment = segment+np.linspace(onset,off,segment.size,dtype=segment.dtype)
        return segment


    def apply_elevation(self, segment: np.ndarray, amplitude: float, sign: [-1,1] = None, randomize: bool = False):
        # Randomize amplitude
        if randomize: amplitude = np.random.rand()*amplitude
        # Copy segment
        segment = np.copy(segment)
        # Randomly choose elevation/depression
        if (sign not in [-1,1]) or (sign is None):
            sign = np.random.choice([-1,1])
        # Compute deviation (cuadratic at the moment)
        linspace = np.linspace(0,np.sqrt(np.abs(amplitude)),segment.size)**2
        deviation = sign*linspace
        # Apply to segment
        segment += deviation.squeeze()
        return segment


    def distribution_draw(self, type: str):
        distribution = self.get_distribution(type)
        if type == 'QRS': 
            amplitude = np.inf
            while (amplitude < self.QRS_ampl_low_thres) or (amplitude > self.QRS_ampl_high_thres):
                amplitude = distribution.rvs()
            amplitude = amplitude.clip(max=1)
        else:
            amplitude = distribution.rvs()
        return amplitude


    def get_distribution(self, type: str):
        if type ==   'P': return self.Pdistribution
        if type ==  'PQ': return self.PQdistribution
        if type == 'QRS': return self.QRSdistribution
        if type ==  'ST': return self.STdistribution
        if type ==   'T': return self.Tdistribution
        if type ==  'TP': return self.TPdistribution


    def get_keys(self, type: str):
        if type ==   'P': return self.Pkeys
        if type ==  'PQ': return self.PQkeys
        if type == 'QRS': return self.QRSkeys
        if type ==  'ST': return self.STkeys
        if type ==   'T': return self.Tkeys
        if type ==  'TP': return self.TPkeys


    def get_waves(self, type: str):
        if type ==   'P': return self.P
        if type ==  'PQ': return self.PQ
        if type == 'QRS': return self.QRS
        if type ==  'ST': return self.ST
        if type ==   'T': return self.T
        if type ==  'TP': return self.TP


    def get_segment_post_function(self, type: str):
        if type ==   'P': return self.P_post_operation
        if type ==  'PQ': return self.PQ_post_operation
        if type == 'QRS': return self.QRS_post_operation
        if type ==  'ST': return self.ST_post_operation
        if type ==   'T': return self.T_post_operation
        if type ==  'TP': return self.TP_post_operation
    
    def P_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        return segment


    def PQ_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        return segment


    def QRS_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if dict_globals['IDs']['ectopics'][index]: segment = self.apply_ectopic_qrs(segment, dict_globals, index)
        return segment


    def ST_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if   dict_globals['has_TV']:    segment = self.random_walk(size=np.random.randint(2,32))
        elif dict_globals['has_tachy']: segment = self.apply_tachy(segment)
        return segment


    def T_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if dict_globals['IDs']['ectopics'][index]: segment = self.apply_ectopic_t(segment, dict_globals, index)
        return segment


    def TP_post_operation(self, segment: np.ndarray, dict_globals: dict, index: int):
        if dict_globals['has_tachy']:   segment = self.apply_tachy(segment)
        return segment


    def apply_tachy(self, segment: np.ndarray):
        new_segment = segment[:np.random.randint(1,self.tachy_maxlen)]
        new_segment = utils.signal.on_off_correction(new_segment)
        if new_segment.ndim == 0:
            new_segment = new_segment[:,None]
        # new_segment = utils.data.ball_scaling(new_segment,metric=self.scaling_metric)
        return new_segment


    def apply_ectopic_qrs(self, segment: np.ndarray, dict_globals: dict, index: int):
        crossings = utils.signal.zero_crossings(segment)[0]
        dict_globals['sign_t'] = -np.sign(utils.signal.signed_maxima(segment[crossings[-2]:crossings[-1]]))
        a_max = np.argmax(np.abs(segment))
        if (crossings[-2] <= a_max) and (crossings[-1] >= a_max):
            sign_elevation = dict_globals['sign_t']
        else:
            sign_elevation = -dict_globals['sign_t']
        segment = self.interpolate_segment(segment,np.random.randint(30,90))
        segment = self.apply_elevation(segment,0.5,sign_elevation)
        return segment


    def apply_ectopic_t(self, segment: np.ndarray, dict_globals: dict, index: int):
        # Only apply if sign w.r.t. qrs is known
        if 'sign_t' in dict_globals:
            # Retrieve sign
            sign_t = dict_globals['sign_t']

            # Check if signs differ; if so, reverse T wave
            segment *= (-1)**(np.sign(utils.signal.signed_maxima(segment)) != sign_t)

            # Delete for future overwriting
            dict_globals.pop('sign_t') # Delete from globals for next ectopic

            # Apply elevation
            segment = self.apply_elevation(segment,0.5,-sign_t)

        # # Enlarge segment's amplitude
        # segment *= 2

        return segment


    def __getitem__(self, i):
        '''Generates one datapoint''' 
        # Generate globals
        dict_globals = self.generate_globals()
        dict_globals['IDs'] = self.generate_IDs(dict_globals)
        dict_globals['amplitudes'] = self.generate_amplitudes(dict_globals)
        dict_globals['elevations'] = self.generate_elevations(dict_globals)
        
        ##### Output data structure
        beats = []
        beat_types = []
        for index in range(self.N): # Unrealistic upper limit
            mark_break = self.generate_cycle(index, dict_globals, beats, beat_types)
            if mark_break: break

        # 5. Registry-wise post-operations
        # 5.0. Trail onsets
        beats = self.trail_onsets(beats)

        # 5.1. Concatenate signal and mask
        signal = np.concatenate(beats)
        masks = self.generate_mask(beats, beat_types, mode_bool=self.labels_as_masks) # Still to be done right
        
        # 5.2. Interpolate signal & mask
        int_len = math.ceil(signal.size*dict_globals['interp_length'])
        signal = self.interpolate_segment(signal,int_len)
        masks = self.interpolate_segment(masks,int_len,axis=-1,kind='next').astype(masks.dtype)

        # 5.3. Apply global amplitude modulation
        signal = signal*dict_globals['global_amplitude']

        # 5.4. Apply whole-signal modifications
        if dict_globals['has_baseline_wander']: signal += self.smooth(self.random_walk(size=signal.size,smoothing_window=self.smoothing_window),self.N//8)
        if dict_globals['has_AF']:              signal = self.apply_AF(signal)
        if dict_globals['has_flatline']:        signal, masks = self.add_flatline(signal, masks, dict_globals)
        
        # 5.5. Apply random onset for avoiding always starting with the same wave at the same location
        on = dict_globals['index_onset']
        signal = signal[on:on+self.N]
        masks = masks[:,on:on+self.N]

        # 5.6. Smooth result
        if self.apply_smoothing: signal = self.smooth(signal, 3)

        # 6. Return
        if self.return_beats: return signal[None,].astype('float32'), masks, beats, beat_types, dict_globals
        else:                 return signal[None,].astype('float32'), masks


    def generate_cycle(self, index: int, dict_globals: dict, beats: list = [], beat_types: list = []):
        # Init output
        total_size = np.sum([beat.size for beat in beats],dtype=int)
        qrs_amplitude = dict_globals['amplitudes']['QRS'][index]

        ### Add all waves ###
        for i,type in enumerate(['P','PQ','QRS','ST','T','TP']):
            # Crop part of the first cycle
            if (index == 0) and (i < dict_globals['begining_wave']): continue

            segment = self.segment_compose(index, type, dict_globals, qrs_amplitude)

            if segment is not None:
                # Add segment to output
                beats.append(segment)
                beat_types.append(type)
                total_size += segment.size

            if total_size >= dict_globals['N']: return True
        return False


    def segment_compose(self, index: int, type: str, dict_globals: dict, qrs_amplitude: float):
        # Retrieve segment information
        if dict_globals['IDs'][type][index] == -1:
            return None
        segment = self.get_segment(type, dict_globals['IDs'][type][index])

        # Segment post-operation
        segment = self.segment_post_operation(type, segment, dict_globals, index)

        # If empty, skip to next
        if segment.size == 0: return

        # Otherwise, apply post-operations
        segment = self.general_post_operation(segment, type, dict_globals, index)

        # Apply amplitude modulation
        segment = self.apply_amplitude(index, segment, type, qrs_amplitude, dict_globals)
        
        return segment


    def get_segment(self, type: str, id: int = None):
        # Get wave information
        waves = self.get_waves(type)
        keys = self.get_keys(type)
        
        # Default id, in case not provided
        if id is None: id = np.random.randint(len(keys))

        # Retrieve segment for modulation
        segment = np.copy(waves[keys[id]])

        return segment


    def segment_post_operation(self, type: str, segment: np.ndarray, dict_globals: dict, index: int):
        post_segment_fnc = self.get_segment_post_function(type)
        
        # Apply wave-specific modifications to the segment
        segment = post_segment_fnc(segment, dict_globals, index)

        return segment


    def add_flatline(self, signal: np.ndarray, masks: np.ndarray, dict_globals: dict) -> Tuple[np.ndarray, np.ndarray]:
        signal = np.pad(signal, (dict_globals['flatline_left'],dict_globals['flatline_right']), mode='edge')
        if self.labels_as_masks:
            masks = np.pad(masks, ((0,0),(dict_globals['flatline_left'],dict_globals['flatline_right'])), mode='constant', constant_values=0)
        else:
            masks = np.pad(masks, (dict_globals['flatline_left'],dict_globals['flatline_right']), mode='constant', constant_values=0)
        return signal, masks


    def apply_amplitude(self, index: int, segment: np.ndarray, type: str, qrs_amplitude: float, dict_globals: dict):
        if type == 'QRS':
            amplitude = qrs_amplitude
        else:
            # Draw from distribution
            amplitude = qrs_amplitude*dict_globals['amplitudes'][type][index] # Apply amplitude on segment

            # Hotfix: conditional to range of QRS amplitude
            if   qrs_amplitude < 0.2: amplitude *= 2.5
            elif qrs_amplitude < 0.4: amplitude *= 1.5

        # Apply per-segment noising
        amplitude *= 0.15*np.random.randn()+1

        # Apply amplitude modulation to segment
        segment *= amplitude
        
        return segment


    def general_post_operation(self, segment: np.ndarray, type: str, dict_globals: dict, index: int):
        # Apply mixup (if applicable)
        if (np.random.rand() < self.proba_mixup) and (type in ['P','QRS','T']):
            segment2 = self.get_segment(type)
            segment2 = self.segment_post_operation(type, segment2, dict_globals, index)
            segment = self.apply_mixup(segment, segment2)

        # Apply interpolation (if applicable, if segment is not empty)
        if (np.random.rand() < self.proba_interpolation) and (segment.size > 1):
            new_size = max(int((segment.size*norm.rvs(1,0.25).clip(min=0.25))),1)
            segment = self.interpolate_segment(segment, new_size)

        # # Apply right extrema elevation/depression
        # if np.random.rand() < self.proba_elevation:
        #     segment = self.apply_elevation(segment, dict_globals['amplitudes'][type][index]/10)
        
        return segment


    def trail_onsets(self, beats: list):
        onset = 0.0
        for i,segment in enumerate(beats):
            segment = self.trail_onset(segment, onset)
            beats[i] = segment
            onset = segment[-1]
        return beats


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


    def apply_mixup(self, segment1: np.ndarray, segment2: np.ndarray):
        if segment1.size != segment2.size:
            segment2 = self.interpolate_segment(segment2, segment1.size)
        (mixup_segment,_) = utils.data.augmentation.mixup(segment1,segment2,self.mixup_alpha,self.mixup_beta)
        mixup_segment = utils.data.ball_scaling(mixup_segment,metric=self.scaling_metric)

        return mixup_segment


    def interpolate_segment(self, y: np.ndarray, new_size: int, **kwargs):
        if y.size != new_size:
            if 'axis' in kwargs:
                size = y.shape[kwargs['axis']]
            else:
                size = y.size
            x_old = np.linspace(0,1,size)
            x_new = np.linspace(0,1,new_size)
            y = interp1d(x_old,y, **kwargs)(x_new)
        return y


    def generate_globals(self):
        dict_globals = {}

        # Probabilities of waves
        dict_globals['same_morph'] = np.random.rand() < self.proba_same_morph
        dict_globals['no_P'] = np.random.rand() < self.proba_no_P
        dict_globals['no_PQ'] = np.random.rand() < self.proba_no_PQ
        dict_globals['no_ST'] = np.random.rand() < self.proba_no_ST
        dict_globals['has_flatline'] = np.random.rand() < self.proba_flatline
        dict_globals['has_TV'] = np.random.rand() < self.proba_TV
        dict_globals['has_AF'] = np.random.rand() < self.proba_AF
        dict_globals['has_tachy'] = np.random.rand() < self.proba_tachy
        dict_globals['has_elevations'] = np.random.rand() < self.proba_elevation
        dict_globals['has_baseline_wander'] = np.random.rand() < self.proba_baseline_wander
        
        # Logical if has TV
        if dict_globals['has_TV']: dict_globals['has_tachy'] = True
        if dict_globals['has_TV']: dict_globals['same_morph'] = True
       
        # Set hyperparameters
        dict_globals['index_onset'] = np.random.randint(50)
        dict_globals['begining_wave'] = np.random.randint(6)
        dict_globals['global_amplitude'] = 1.+(np.random.randn()*self.amplitude_std)
        dict_globals['interp_length'] = max([1.+(np.random.randn()*self.interp_std),0.5])
        if dict_globals['has_flatline']: dict_globals['flatline_length'] = np.random.randint(1,self.N//np.random.randint(2,5))
        if dict_globals['has_flatline']: dict_globals['flatline_left'] = np.random.randint(dict_globals['flatline_length'])
        if dict_globals['has_flatline']: dict_globals['flatline_right'] = dict_globals['flatline_length']-dict_globals['flatline_left']
        dict_globals['N'] = math.ceil((self.N+dict_globals['index_onset']-dict_globals.get('flatline_length',0))/dict_globals['interp_length'])

        return dict_globals


    def generate_IDs(self, dict_globals: dict):
        IDs = {}

        ##### Identifiers
        if dict_globals['same_morph']:
            IDs[  'P'] = np.array([np.random.randint(len(  self.P))]*self.cycles)
            IDs[ 'PQ'] = np.array([np.random.randint(len( self.PQ))]*self.cycles)
            IDs['QRS'] = np.array([np.random.randint(len(self.QRS))]*self.cycles)
            IDs[ 'ST'] = np.array([np.random.randint(len( self.ST))]*self.cycles)
            IDs[  'T'] = np.array([np.random.randint(len(  self.T))]*self.cycles)
            IDs[ 'TP'] = np.array([np.random.randint(len( self.TP))]*self.cycles)
        else:
            IDs[  'P'] = np.random.randint(len(  self.P), size=self.cycles)
            IDs[ 'PQ'] = np.random.randint(len( self.PQ), size=self.cycles)
            IDs['QRS'] = np.random.randint(len(self.QRS), size=self.cycles)
            IDs[ 'ST'] = np.random.randint(len( self.ST), size=self.cycles)
            IDs[  'T'] = np.random.randint(len(  self.T), size=self.cycles)
            IDs[ 'TP'] = np.random.randint(len( self.TP), size=self.cycles)

        if dict_globals['has_AF']:
            IDs['P'] = np.repeat(np.random.randint(len(self.P)),self.cycles)

        # In case QRS is not expressed
        filt_QRS = np.random.rand(self.cycles) > (1-self.proba_no_QRS)

        # Exceptions according to different conditions
        IDs[  'P'][           (np.random.rand(self.cycles) < self.proba_no_P)  | dict_globals['no_P']  | dict_globals['has_TV']] = -1
        IDs[ 'PQ'][filt_QRS | (np.random.rand(self.cycles) < self.proba_no_PQ) | dict_globals['no_PQ'] | dict_globals['has_TV']] = -1
        IDs['QRS'][filt_QRS] = -1
        IDs[ 'ST'][filt_QRS | (np.random.rand(self.cycles) < self.proba_no_ST) | dict_globals['no_ST']] = -1
        IDs[  'T'][filt_QRS] = -1
        IDs[ 'TP'][np.full((self.cycles),dict_globals['has_TV'],dtype=bool)] = -1

        # Generate ectopics
        IDs['ectopics'] = np.random.rand(self.cycles) < self.proba_ectopics
        # Logical if has TV
        if dict_globals['has_TV']: IDs['ectopics'] = np.ones_like(IDs['ectopics'],dtype=bool)
        IDs['P'][IDs['ectopics']] = -1
        IDs['ST'][IDs['ectopics']] = -1

        return IDs


    def generate_amplitudes(self, dict_globals: dict):
        amplitudes = {
            'P'   : self.Pdistribution.rvs(self.cycles),
            'PQ'  : self.PQdistribution.rvs(self.cycles),
            'QRS' : self.QRSdistribution.rvs(self.cycles),
            'ST'  : self.STdistribution.rvs(self.cycles),
            'T'   : self.Tdistribution.rvs(self.cycles),
            'TP'  : self.TPdistribution.rvs(self.cycles),
        }

        # QRS in case low/high voltage
        filter = (amplitudes['QRS'] < self.QRS_ampl_low_thres) | (amplitudes['QRS'] > self.QRS_ampl_high_thres)
        while np.any(filter):
            # Retrieve generous sample, faster than sampling twice
            new_amplitudes = self.QRSdistribution.rvs(self.cycles)
            new_amplitudes = new_amplitudes[(new_amplitudes >= self.QRS_ampl_low_thres) | (new_amplitudes <= self.QRS_ampl_high_thres)]
            # Pad/crop the new amplitudes
            pad_len = filter.sum()-new_amplitudes.size
            if   pad_len < 0: new_amplitudes = new_amplitudes[:filter.sum()]
            elif pad_len > 0: new_amplitudes = np.pad(new_amplitudes,(0,pad_len))
            # Input into the amplitudes vector
            amplitudes['QRS'][filter] = new_amplitudes
            filter = (amplitudes['QRS'] < self.QRS_ampl_low_thres) | (amplitudes['QRS'] > self.QRS_ampl_high_thres)
        amplitudes['QRS'] = amplitudes['QRS'].clip(max=1)

        # T wave in case ectopics
        filter = (amplitudes['T'] < self.ectopic_amplitude_threshold) & dict_globals['IDs']['ectopics']
        while np.any(filter):
            # Retrieve generous sample, faster than sampling twice
            new_amplitudes = self.Tdistribution.rvs(self.cycles)
            new_amplitudes = new_amplitudes[new_amplitudes >= self.ectopic_amplitude_threshold]
            # Pad/crop the new amplitudes
            pad_len = filter.sum()-new_amplitudes.size
            if   pad_len < 0: new_amplitudes = new_amplitudes[:filter.sum()]
            elif pad_len > 0: new_amplitudes = np.pad(new_amplitudes,(0,pad_len))
            # Input into the amplitudes vector
            amplitudes['T'][filter] = new_amplitudes
            filter = (amplitudes['T'] < self.ectopic_amplitude_threshold) & dict_globals['IDs']['ectopics']

        return amplitudes


    def generate_elevations(self, dict_globals: dict):
        elevations = {}

        # Generate elevation template
        elevation_template = np.random.rand(self.cycles,6)

        # Refine template's results - zero if the ID is zero
        filt = np.vstack((dict_globals['IDs']['P'],   dict_globals['IDs']['PQ'],
                        dict_globals['IDs']['QRS'], dict_globals['IDs']['ST'],
                        dict_globals['IDs']['T'],   dict_globals['IDs']['TP'])).T == -1
        elevation_template[filt] = 0

        # Define correction factor for number of non-negative amplitudes
        num_elements = np.logical_not(filt).sum(-1,keepdims=True)
        correction_factor = 1/(np.ones_like(filt,dtype=int)*num_elements)
        correction_factor[filt] = 0

        # Define % elevation per active segment
        elevation_template = elevation_template/elevation_template.sum(-1,keepdims=True) - correction_factor

        elevations['P']   = elevation_template[:,0]
        elevations['PQ']  = elevation_template[:,1]
        elevations['QRS'] = elevation_template[:,2]
        elevations['ST']  = elevation_template[:,3]
        elevations['T']   = elevation_template[:,4]
        elevations['TP']  = elevation_template[:,5]

        return elevations


    def apply_AF(self, signal: np.ndarray):
        # select random P wave as template
        pAF = self.get_segment('P')
        N = signal.size

        # Mirror on negative
        sign = np.random.choice([-1,1])
        template = [sign*pAF[:-1],-sign*pAF[:-1]]

        # Repeat to reach N samples
        template = np.concatenate(template*math.ceil(N/sum([template[0].size,template[1].size])))

        # Crop to size N 
        if template.size != N:
            onset = np.random.randint(template.size-N)
            template = template[onset:onset+N]

        # Interpolate on x axis to make less "stiff"
        x = np.random.rand(N-1)*(np.random.randint(5,size=N-1)**np.random.randint(5,size=N-1))
        x = np.cumsum(self.smooth(np.hstack(([0.],x)),N//16))
        x = x/np.max(x)
        template = interp1d(np.linspace(0,1,N),template)(x)

        # Apply noise 
        template = template*np.random.randn(template.size)
        template = self.smooth(template,pAF.size//2)

        # Sample amplitude > 0.3 (to make any noticeable difference on the signal)
        amplitude = self.Pdistribution.rvs()
        while (amplitude < 0.3) and (amplitude > 0.6): amplitude = self.Pdistribution.rvs()

        # Apply AF on signal
        signal = signal+template*amplitude
        
        return signal

