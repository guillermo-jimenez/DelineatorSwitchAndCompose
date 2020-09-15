from typing import Callable
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
                 proba_TV = 0.05, proba_AF = 0.05, proba_tachy = 0.05, 
                 tachy_maxlen = 10, add_baseline_wander = True, 
                 amplitude_std = 0.25, interp_std = 0.25,
                 smoothing_window = 51, labels_as_masks = True, 
                 QRS_ampl_low_thres = 0.1, QRS_ampl_high_thres = 1.15,
                 scaling_metric: Callable = utils.signal.amplitude,
                 return_beats: bool = False):
        #### Segments ####
        # P wave
        self.P = P
        self.Pdistribution = Pdistribution
        self.Pkeys = list(P.keys())
        # PQ wave
        self.PQ = PQ
        self.PQdistribution = PQdistribution
        self.PQkeys = list(PQ.keys())
        # QRS wave
        self.QRS = QRS
        self.QRSkeys = list(QRS.keys())
        self.QRSdistribution = QRSdistribution
        # ST wave
        self.ST = ST
        self.STdistribution = STdistribution
        self.STkeys = list(ST.keys())
        # T wave
        self.T = T
        self.Tdistribution = Tdistribution
        self.Tkeys = list(T.keys())
        # TP wave
        self.TP = TP
        self.TPdistribution = TPdistribution
        self.TPkeys = list(TP.keys())
        
        #### Generation hyperparams ####
        # Ints'n'stuff
        self.length = length
        self.N = N
        self.noise = noise
        self.add_baseline_wander = add_baseline_wander
        self.smoothing_window = smoothing_window
        self.labels_as_masks = labels_as_masks
        self.amplitude_std = amplitude_std
        self.interp_std = interp_std
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.tachy_maxlen = tachy_maxlen
        self.return_beats = return_beats
        self.scaling_metric = scaling_metric
        self.QRS_ampl_low_thres = QRS_ampl_low_thres
        self.QRS_ampl_high_thres = QRS_ampl_high_thres
        # Probabilities
        self.proba_no_P = proba_no_P
        self.proba_no_QRS = proba_no_QRS
        self.proba_no_PQ = proba_no_PQ
        self.proba_no_ST = proba_no_ST
        self.proba_TV = proba_TV
        self.proba_AF = proba_AF
        self.proba_same_morph = proba_same_morph
        self.proba_elevation = proba_elevation
        self.proba_interpolation = proba_interpolation
        self.proba_mixup = proba_mixup
        self.proba_tachy = proba_tachy
        
    def __len__(self):
        '''Denotes the number of elements in the dataset'''
        return self.length

    def __random_walk(self, scale: float = 0.01**(2*0.5), size: int = 2048, smoothing_window: int = None, conv_mode: str = 'same'):
        noise = np.cumsum(norm.rvs(scale=scale,size=size))
        if smoothing_window is not None:
            window = np.hamming(smoothing_window)/(smoothing_window//2)
            noise = np.convolve(noise, window, mode=conv_mode)
        return noise

    def __trail_onset(self,segment,onset):
        onset = onset-segment[0]
        off = onset-segment[0]+segment[-1]
        segment = segment+np.linspace(onset,off,segment.size,dtype=segment.dtype)
        return segment

    def __apply_convolution(self, segment: np.ndarray, template: np.ndarray):
        # 1.2.3. Common operations
        sign = np.random.choice([0,1])
        mirrored_template = np.concatenate([((-1)**(sign))*template,((-1)**(sign+1))*template])
        segment = np.convolve(segment, mirrored_template)
        segment = utils.data.ball_scaling(segment,metric=self.scaling_metric)
        return segment
        
    def __apply_elevation(self, segment: np.ndarray, type: str):
        # Retrieve amplitude
        right_amplitude = self.distribution_draw(type)*0.05
        # Randomly choose elevation/depression
        sign = np.random.choice([-1,1])
        # Compute deviation (cuadratic at the moment)
        linspace = np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2
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
        if   type == 'P':   return self.Pdistribution
        elif type == 'PQ':  return self.PQdistribution
        elif type == 'QRS': return self.QRSdistribution
        elif type == 'ST':  return self.STdistribution
        elif type == 'T':   return self.Tdistribution
        elif type == 'TP':  return self.TPdistribution

    def get_keys(self, type: str):
        if   type == 'P':   return self.Pkeys
        elif type == 'PQ':  return self.PQkeys
        elif type == 'QRS': return self.QRSkeys
        elif type == 'ST':  return self.STkeys
        elif type == 'T':   return self.Tkeys
        elif type == 'TP':  return self.TPkeys

    def get_waves(self, type: str):
        if   type == 'P':   return self.P
        elif type == 'PQ':  return self.PQ
        elif type == 'QRS': return self.QRS
        elif type == 'ST':  return self.ST
        elif type == 'T':   return self.T
        elif type == 'TP':  return self.TP

    def __get_pre_function(self, type: str):
        if   type == 'P':   return self.__pre_P
        elif type == 'PQ':  return self.__pre_PQ
        elif type == 'QRS': return self.__pre_QRS
        elif type == 'ST':  return self.__pre_ST
        elif type == 'T':   return self.__pre_T
        elif type == 'TP':  return self.__pre_TP

    def __pre_P(self, segment: np.ndarray, dict_globals: dict, template: np.ndarray):
        if dict_globals['has_AF']:
            # Segment as smoothed random noise
            segment = np.random.randn(2*segment.size)
            segment = np.convolve(segment,np.hamming(segment.size)/(segment.size//2),mode='same')[segment.size//4:-segment.size//4]
            # Convolve with template
            segment = self.__apply_convolution(segment, template)
            # discard extra information
            on = np.random.choice([0,segment.size//4,segment.size//2])
            segment = utils.signal.on_off_correction(segment[on:on+segment.size//2])
        return segment

    def __pre_PQ(self, segment: np.ndarray, dict_globals: dict, template: np.ndarray):
        return segment

    def __pre_QRS(self, segment: np.ndarray, dict_globals: dict, template: np.ndarray):
        return segment

    def __pre_ST(self, segment: np.ndarray, dict_globals: dict, template: np.ndarray):
        if dict_globals['has_TV']:      segment = self.__random_walk(size=np.random.randint(2,32))
        elif dict_globals['has_tachy']: segment = self.__apply_tachy(segment)
        return segment

    def __pre_T(self, segment: np.ndarray, dict_globals: dict, template: np.ndarray):
        return segment

    def __pre_TP(self, segment: np.ndarray, dict_globals: dict, template: np.ndarray):
        if dict_globals['has_tachy']:   segment = self.__apply_tachy(segment)
        if dict_globals['has_AF']:      segment = self.__apply_convolution(segment, template)
        return segment

    def __apply_tachy(self, segment: np.ndarray):
        new_segment = segment[:np.random.randint(1,self.tachy_maxlen)]
        new_segment = utils.signal.on_off_correction(new_segment)
        if new_segment.ndim == 0:
            new_segment = new_segment[:,None]
        # new_segment = utils.data.ball_scaling(new_segment,metric=self.scaling_metric)
        return new_segment

    def __generate_segment(self, id: int, type: str, dict_globals: dict, onset: int, template: np.ndarray = None):
        # 0. Get wave information
        waves = self.get_waves(type)
        keys = self.get_keys(type)
        pre_fnc = self.__get_pre_function(type)
        
        # 1. Retrieve segment for modulation
        segment = waves[keys[id]].copy()

        # 2. Apply mixup to segment (if applicable)
        if (np.random.rand(1) < self.proba_mixup) and (type in ['P','QRS','T']):
            second_segment = waves[keys[np.random.randint(len(keys))]]
            if segment.size != second_segment.size:
                second_segment = interp1d(np.linspace(0,1,second_segment.size),second_segment)(np.linspace(0,1,segment.size))
            (segment,_) = utils.data.augmentation.mixup(segment,second_segment,self.mixup_alpha,self.mixup_beta)
            segment = utils.data.ball_scaling(segment,metric=self.scaling_metric)

        # 3. Apply wave_specific modifications to the segment
        segment = pre_fnc(segment, dict_globals, template)

        # Case segment is not empty
        if segment.size > 0:
            ####################################################################################
            # 2. Apply amplitude modulation
            segment *= self.distribution_draw(type) # Apply amplitude on segment

            # 3. Per-segment amplitude noising
            noise = 0.15*np.random.randn(1)+1
            segment *= noise
            
            # 4. Apply interpolation (if applicable)
            if (np.random.rand(1) < self.proba_interpolation) and (segment.size > 1):
                new_size = max(int((segment.size*norm.rvs(1,0.25).clip(min=0.25))),1)
                x = np.linspace(0,1,segment.size)
                x_new = np.linspace(0,1,new_size)
                segment = interp1d(x,segment)(x_new)
            
            # 5. Per-segment right extrema elevation/depression
            if np.random.rand(1) < self.proba_elevation:
                segment = self.__apply_elevation(segment, type)
            
            # 6. Onset trailing
            segment = self.__trail_onset(segment, onset)

        # 7. Final segment storage
        mask_value = 1 if type == 'P' else 2 if type == 'QRS' else 3 if type == 'T' else 0
        mask = np.full(segment.shape,mask_value,dtype='int8')

        return segment, mask

    def __get_cycle(self, index: int, onset: float, dict_globals: dict, dict_IDs: dict, templates: dict = {}, record_size: int = None):
        # Init output
        beats = []
        masks = []
        mark_break = False
        if record_size is None: record_size = 0

        ### Add all waves ###
        types = ['P', 'PQ', 'QRS', 'ST', 'T', 'TP']
        for i in range(len(types)):
            if (index == 0) and (i < dict_globals['begining_wave']):
                continue
                
            # Define type
            type = types[i]

            # Get ID for this wave and template (if any)
            id = dict_IDs[type][i]
            template = templates.get(type,None)

            # Retrieve segment information
            seg,msk = self.__generate_segment(id, type, dict_globals, onset, template)

            # Add segment to output
            beats.append(seg)
            masks.append(msk)
            record_size += seg.size
            onset = beats[-1][-1]

            if (math.floor(record_size*dict_globals['interp_length'])-dict_globals['index_onset']) >= self.N:
                mark_break = True
                break

        return beats, masks, record_size, onset, mark_break


    def __getitem__(self, i: int):
        '''Generates one datapoint''' 
        # Store globals
        dict_globals = {}
        dict_IDs = {}
        templates = {}

        # Set hyperparameters
        dict_globals['index_onset'] = np.random.randint(50)
        dict_globals['begining_wave'] = np.random.randint(6)
        dict_globals['global_amplitude'] = 1.+(np.random.randn(1)*self.amplitude_std)
        dict_globals['interp_length'] = max([1.+(np.random.randn(1)*self.interp_std),0.5])

        # Probabilities of waves
        dict_globals['does_not_have_P'] = np.random.rand(1) < self.proba_no_P
        dict_globals['does_not_have_PQ'] = np.random.rand(1) < self.proba_no_PQ
        dict_globals['does_not_have_ST'] = np.random.rand(1) < self.proba_no_ST
        dict_globals['has_TV'] = np.random.rand(1) < self.proba_TV
        dict_globals['has_AF'] = np.random.rand(1) < self.proba_AF
        dict_globals['has_tachy'] = np.random.rand(1) < self.proba_tachy

        ##### Identifiers
        if np.random.rand(1) < self.proba_same_morph:
            dict_IDs['P'] = np.array([np.random.randint(len(self.P))]*self.N)
            dict_IDs['PQ'] = np.array([np.random.randint(len(self.PQ))]*self.N)
            dict_IDs['QRS'] = np.array([np.random.randint(len(self.QRS))]*self.N)
            dict_IDs['ST'] = np.array([np.random.randint(len(self.ST))]*self.N)
            dict_IDs['T'] = np.array([np.random.randint(len(self.T))]*self.N)
            dict_IDs['TP'] = np.array([np.random.randint(len(self.TP))]*self.N)
        else:
            dict_IDs['P'] = np.random.randint(len(self.P),size=self.N)
            dict_IDs['PQ'] = np.random.randint(len(self.PQ),size=self.N)
            dict_IDs['QRS'] = np.random.randint(len(self.QRS),size=self.N)
            dict_IDs['ST'] = np.random.randint(len(self.ST),size=self.N)
            dict_IDs['T'] = np.random.randint(len(self.T),size=self.N)
            dict_IDs['TP'] = np.random.randint(len(self.TP),size=self.N)

        if dict_globals['has_AF']:
            dict_IDs['P'] = np.repeat(np.random.randint(len(self.P),size=1),self.N)
            pAF = self.P[self.Pkeys[dict_IDs['P'][0]]]
            pAF = interp1d(np.linspace(0,1,pAF.size),pAF)(np.linspace(0,1,pAF.size//2))
            pdist = 4*self.Pdistribution.rvs()
            templates['P'] = pdist*pAF
            templates['TP'] = pdist*pAF
            
        # In case QRS is not expressed
        filt_QRS = np.random.rand(self.N) > (1-self.proba_no_QRS)

        # P wave
        dict_IDs['P'][(np.random.rand(self.N) < self.proba_no_P) | dict_globals['does_not_have_P'] | dict_globals['has_TV']] = -1
        dict_IDs['PQ'][filt_QRS | (np.random.rand(self.N) < self.proba_no_PQ) | dict_globals['does_not_have_PQ'] | dict_globals['has_TV']] = -1
        dict_IDs['QRS'][filt_QRS] = -1
        dict_IDs['ST'][filt_QRS | (np.random.rand(self.N) < self.proba_no_ST) | dict_globals['does_not_have_ST']] = -1
        dict_IDs['T'][filt_QRS] = -1
        dict_IDs['TP'][np.full((self.N),dict_globals['has_TV'],dtype=bool)] = -1
        
        ##### Output data structure
        beats = []
        masks = []
        record_size = 0
        onset = 0.0
        for i in range(self.N): # Unrealistic upper limit
            bts, msk, record_size, onset, mark_break = self.__get_cycle(i, onset, dict_globals, dict_IDs, templates, record_size)
            beats += bts
            masks += msk
            if mark_break:
                break

        # 5. Registry-wise post-operations
        # 5.1. Concatenate signal and mask
        signal = np.concatenate(beats)
        masks = np.concatenate(masks)

        # 5.2. Interpolate signal & mask
        x = np.linspace(0,1,signal.size)
        signal = interp1d(x,signal)(np.linspace(0,1,math.ceil(signal.size*dict_globals['interp_length'])))
        masks = interp1d(x,masks,kind='next')(np.linspace(0,1,math.ceil(masks.size*dict_globals['interp_length']))).astype(int)

        # 5.3. Apply random onset for avoiding always starting with the same wave at the same location
        signal = signal[dict_globals['index_onset']:dict_globals['index_onset']+self.N]
        masks = masks[dict_globals['index_onset']:dict_globals['index_onset']+self.N]

        # 5.4. Apply global amplitude modulation
        signal = signal*dict_globals['global_amplitude']

        # 5.5. Express as masks (optional)
        if self.labels_as_masks:
            masks_all = np.zeros((3,self.N),dtype=bool)
            masks_all[0,:] = (masks == 1)
            masks_all[1,:] = (masks == 2)
            masks_all[2,:] = (masks == 3)
        else:
            masks_all = masks.astype('int32')

        # 5.6. Add baseline wander
        if self.add_baseline_wander:
            signal = signal + self.__random_walk(smoothing_window=self.smoothing_window)
        
        if self.return_beats:
            return signal[None,].astype('float32'), masks_all, beats, masks
        else:
            return signal[None,].astype('float32'), masks_all
      



    # def __getitem__(self, i: int):
    #     '''Generates one datapoint''' 
    #     # Set hyperparameters
    #     onset = np.random.randint(50)
    #     begining_wave = np.random.randint(6)
    #     global_amplitude = 1.+(np.random.randn(1)*self.amplitude_std)
    #     interp_length = max([1.+(np.random.randn(1)*self.interp_std),0.5])

    #     # Probabilities of waves
    #     does_not_have_P = np.random.rand(1) > (1-self.proba_no_P)
    #     does_not_have_PQ = np.random.rand(1) > (1-self.proba_no_PQ)
    #     does_not_have_ST = np.random.rand(1) > (1-self.proba_no_ST)
    #     has_TV = np.random.rand(1) > (1-self.proba_TV)
    #     has_AF = np.random.rand(1) > (1-self.proba_AF)
    #     has_same_morph = np.random.rand(1) > (1-self.proba_same_morph)
    #     has_tachy = np.random.rand(1) > (1-self.proba_tachy)
    #     has_elevation = np.random.rand(self.N) > (1-self.proba_elevation)
    #     has_interpolation = np.random.rand(self.N) > (1-self.proba_interpolation)
    #     has_mixup = np.random.rand(self.N) > (1-self.proba_mixup)

    #     ##### Identifiers
    #     if has_same_morph:
    #         id_P = np.repeat(np.random.randint(len(self.P),size=1),self.N)
    #         id_PQ = np.repeat(np.random.randint(len(self.PQ),size=1),self.N)
    #         id_QRS = np.repeat(np.random.randint(len(self.QRS),size=1),self.N)
    #         id_ST = np.repeat(np.random.randint(len(self.ST),size=1),self.N)
    #         id_T = np.repeat(np.random.randint(len(self.T),size=1),self.N)
    #         id_TP = np.repeat(np.random.randint(len(self.TP),size=1),self.N)
    #     else:
    #         id_P = np.random.randint(len(self.P),size=self.N)
    #         id_PQ = np.random.randint(len(self.PQ),size=self.N)
    #         id_QRS = np.random.randint(len(self.QRS),size=self.N)
    #         id_ST = np.random.randint(len(self.ST),size=self.N)
    #         id_T = np.random.randint(len(self.T),size=self.N)
    #         id_TP = np.random.randint(len(self.TP),size=self.N)

    #     if has_AF:
    #         id_P = np.repeat(np.random.randint(len(self.P),size=1),self.N)
    #         pAF = self.P[self.Pkeys[id_P[0]]]
    #         pAF = interp1d(np.linspace(0,1,pAF.size),pAF)(np.linspace(0,1,pAF.size//2))
    #         pdist = 4*self.Pdistribution.rvs()
            
    #     # In case QRS is not expressed
    #     filt_QRS = np.random.rand(self.N) > (1-self.proba_no_QRS)

    #     # P wave
    #     id_P[(np.random.rand(self.N) < self.proba_no_P) | does_not_have_P | has_TV] = -1
    #     id_PQ[filt_QRS | (np.random.rand(self.N) < self.proba_no_PQ) | does_not_have_PQ | has_TV] = -1
    #     id_QRS[filt_QRS] = -1
    #     id_ST[filt_QRS | (np.random.rand(self.N) < self.proba_no_ST) | does_not_have_ST] = -1
    #     id_T[filt_QRS] = -1
    #     id_TP[np.full((self.N),has_TV,dtype=bool)] = -1
        
    #     ##### Output data structure
    #     beats = []
    #     masks = []
    #     record_size = 0
    #     mark_break = False
    #     for i in range(self.N): # Unrealistic upper limit
    #         for j in range(6):
    #             if (i == 0) and (j < begining_wave): continue
    #             if j==0: id,keys,waves,distribution = id_P[i],   self.Pkeys,   self.P,   self.Pdistribution
    #             if j==1: id,keys,waves,distribution = id_PQ[i],  self.PQkeys,  self.PQ,  self.PQdistribution
    #             if j==2: id,keys,waves,distribution = id_QRS[i], self.QRSkeys, self.QRS, self.QRSdistribution
    #             if j==3: id,keys,waves,distribution = id_ST[i],  self.STkeys,  self.ST,  self.STdistribution
    #             if j==4: id,keys,waves,distribution = id_T[i],   self.Tkeys,   self.T,   self.Tdistribution
    #             if j==5: id,keys,waves,distribution = id_TP[i],  self.TPkeys,  self.TP,  self.TPdistribution
    #             if (math.floor(record_size*interp_length)-onset) >= self.N: 
    #                 mark_break = True
    #                 break
                
    #             # 0. Skip beats if not wanted, according to above rules
    #             if id == -1: continue # Flag for skipping unwanted beat
                
    #             # 1. Retrieve segment for modulation
    #             segment = waves[keys[id]].copy()

    #             # 1.1. If selected, apply mixup to segment
    #             if has_mixup[len(beats)] and j in [0,2,4]:
    #                 segment = self.__apply_mixup(segment, keys, waves, distribution)
    #             if has_tachy and j in [3,5]:
    #                 segment = segment[:np.random.randint(self.tachy_maxlen)]
    #                 if segment.size < 2:
    #                     continue
    #                 segment = utils.signal.on_off_correction(segment)
    #                 segment = utils.data.ball_scaling(segment,metric=self.scaling_metric)

    #             # 1.2. If selected, substitute ST segment by random walk
    #             if (j==3) and has_TV:
    #                 nx = np.random.randint(32)
    #                 if nx < 2: continue # Avoid always having a TV with some space between QRS and T
    #                 segment = np.convolve(np.cumsum(norm.rvs(scale=0.01**(2*0.5),size=nx)),np.hamming(nx)/(nx//2),mode='same')

    #             # 1.2. If selected, convolve TP segment/random noise by P wave for AF simulation
    #             if has_AF and j in [0,5]:
    #                 # 1.2.1. Case is P wave
    #                 if j==0:
    #                     segment = np.random.randn(waves[keys[id]].size)
    #                     segment = np.convolve(segment,np.hamming(segment.size)/(segment.size//2),mode='same')[segment.size//4:-segment.size//4]
    #                 # 1.2.2. Case is TP segment
    #                 if j==5:
    #                     segment = waves[keys[id]].copy()
                    
    #                 # 1.2.3. Common operations
    #                 sign = np.random.choice([0,1])
    #                 segment = np.convolve(segment,np.concatenate([((-1)**(sign))*pAF,((-1)**(sign+1))*pAF]))
    #                 segment = utils.data.ball_scaling(segment,metric=self.scaling_metric)
                    
    #                 # 1.2.4. Random cropping for P wave
    #                 if j==0:
    #                     on = np.random.choice([0,segment.size//4,segment.size//2])
    #                     segment = utils.signal.on_off_correction(segment[on:on+segment.size//2])

    #             # 2. Apply amplitude modulation
    #             # 2.1. Case has AF
    #             if has_AF and (j in [0,5]):
    #                 amplitude = pdist
    #             else:
    #                 amplitude = distribution.rvs()

    #             # 2.N. Apply the amplitude modulation on the segment
    #             segment *= amplitude
                
    #             # 3. Apply segment-wise augmentations.
    #             # 3.1. Per-segment amplitude noising
    #             noise = 0.15*np.random.randn(1)+1
    #             segment *= noise
                
    #             # 3.2. Per-segment interpolation
    #             if has_interpolation[len(beats)]:
    #                 x = np.linspace(0,1,segment.size)
    #                 x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.25))))
    #                 segment = interp1d(x,segment)(x_new)
                
    #             # 3.3. Per-segment right extrema elevation/depression
    #             if has_elevation[len(beats)]:
    #                 right_amplitude = distribution.rvs()*0.05
    #                 segment += np.random.choice([-1,1])*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                
    #             # 4. Segment-wise post-operations
    #             # 4.1. Onset trailing for coherency
    #             segment = self.__trail_onset(segment,beats[-1][-1] if len(beats) != 0 else 0)

    #             # 4.1. Final segment storage
    #             mask_value = 1 if j==0 else 2 if j==2 else 3 if j==4 else 0
    #             beats.append(segment)
    #             masks.append(np.full(segment.shape,mask_value,dtype='int8'))
    #             record_size += segment.size
    #         if mark_break:
    #             break

    #     # 5. Registry-wise post-operations
    #     # 5.1. Concatenate signal and mask
    #     signal = np.concatenate(beats)
    #     masks = np.concatenate(masks)

    #     # 5.2. Interpolate signal & mask
    #     x = np.linspace(0,1,signal.size)
    #     signal = interp1d(x,signal)(np.linspace(0,1,math.ceil(signal.size*interp_length)))
    #     masks = interp1d(x,masks,kind='next')(np.linspace(0,1,math.ceil(masks.size*interp_length))).astype(int)

    #     # 5.3. Apply random onset for avoiding always starting with the same wave at the same location
    #     signal = signal[onset:onset+self.N]
    #     masks = masks[onset:onset+self.N]

    #     # 5.4. Apply global amplitude modulation
    #     signal = signal*global_amplitude

    #     # 5.5. Express as masks (optional)
    #     if self.labels_as_masks:
    #         masks_all = np.zeros((3,self.N),dtype=bool)
    #         masks_all[0,:] = (masks == 1)
    #         masks_all[1,:] = (masks == 2)
    #         masks_all[2,:] = (masks == 3)
    #     else:
    #         masks_all = masks.astype('int32')

    #     # 5.6. Add baseline wander
    #     if self.add_baseline_wander:
    #         signal = signal + np.convolve(np.cumsum(norm.rvs(scale=0.01**(2*0.5),size=self.N)),np.hamming(self.smoothing_window)/(self.smoothing_window/2),mode='same')
        
    #     return signal[None,].astype('float32'), masks_all
