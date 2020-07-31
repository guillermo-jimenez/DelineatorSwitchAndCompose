import torch
import torch.utils
import numpy as np
import math
from scipy.stats import norm
from scipy.interpolate import interp1d


def trailonset(sig,on):
    on = on-sig[0]
    off = on-sig[0]+sig[-1]
    sig = sig+np.linspace(on,off,sig.size,dtype=sig.dtype)
    
    return sig


class Dataset(torch.utils.data.Dataset):
    '''Generates data for PyTorch'''

    def __init__(self, P, QRS, T, PQ, ST, TP, 
                 Pamplitudes, QRSamplitudes, Tamplitudes, 
                 PQamplitudes, STamplitudes, TPamplitudes, 
                 length,N = 2048, noise = 0.005, proba_no_P = 0.25,
                 proba_no_QRS = 0.01, proba_no_PQ = 0.15, 
                 proba_no_ST = 0.15, proba_same_morph = 0.2,
                 proba_elevation = 0.2, proba_interpolation = 0.2,
                 proba_TV = 0.05, add_baseline_wander = True, 
                 amplitude_std = 0.25, interp_std = 0.25,
                 window = 51, labels_as_masks = True):
        # Segments
        self.P = P
        self.QRS = QRS
        self.T = T
        self.PQ = PQ
        self.ST = ST
        self.TP = TP
        self.Pamplitudes = Pamplitudes
        self.QRSamplitudes = QRSamplitudes
        self.Tamplitudes = Tamplitudes
        self.PQamplitudes = PQamplitudes
        self.STamplitudes = STamplitudes
        self.TPamplitudes = TPamplitudes
        self.Pkeys = list(P.keys())
        self.QRSkeys = list(QRS.keys())
        self.Tkeys = list(T.keys())
        self.PQkeys = list(PQ.keys())
        self.STkeys = list(ST.keys())
        self.TPkeys = list(TP.keys())
        
        # Generation hyperparams
        self.length = length
        self.N = N
        self.noise = noise
        self.add_baseline_wander = add_baseline_wander
        self.window = window
        self.labels_as_masks = labels_as_masks
        self.amplitude_std = amplitude_std
        self.interp_std = interp_std

        # Probabilities
        self.proba_no_P = proba_no_P
        self.proba_no_QRS = proba_no_QRS
        self.proba_no_PQ = proba_no_PQ
        self.proba_no_ST = proba_no_ST
        self.proba_TV = proba_TV
        self.proba_same_morph = proba_same_morph
        self.proba_elevation = proba_elevation
        self.proba_interpolation = proba_interpolation
        
        
    def __len__(self):
        '''Denotes the number of elements in the dataset'''
        return self.length

    def __getitem__(self, i: int):
        '''Generates one datapoint''' 
        # Set hyperparameters
        onset = np.random.randint(0,50)
        begining_wave = np.random.randint(0,6)
        global_amplitude = 1.+(np.random.randn(1)*self.amplitude_std)
        interp_length = max([1.+(np.random.randn(1)*self.interp_std),0.2])

        # Probabilities of waves
        does_not_have_P = (np.random.rand(1) > (1-self.proba_no_P))
        does_not_have_PQ = (np.random.rand(1) > (1-self.proba_no_PQ))
        does_not_have_ST = (np.random.rand(1) > (1-self.proba_no_ST))
        has_TV = (np.random.rand(1) > (1-self.proba_TV))
        has_same_morph = (np.random.rand(1) > (1-self.proba_same_morph))
        has_elevation = (np.random.rand(self.N) > (1-self.proba_elevation))
        has_interpolation = (np.random.rand(self.N) > (1-self.proba_interpolation))

        ##### Data structure
        ids = []

        ##### Identifiers
        if not has_same_morph:
            id_P = np.random.randint(0,len(self.P),size=self.N)
            id_PQ = np.random.randint(0,len(self.PQ),size=self.N)
            id_QRS = np.random.randint(0,len(self.QRS),size=self.N)
            id_ST = np.random.randint(0,len(self.ST),size=self.N)
            id_T = np.random.randint(0,len(self.T),size=self.N)
            id_TP = np.random.randint(0,len(self.TP),size=self.N)
        else:
            id_P = np.repeat(np.random.randint(0,len(self.P),size=1),self.N)
            id_PQ = np.repeat(np.random.randint(0,len(self.PQ),size=1),self.N)
            id_QRS = np.repeat(np.random.randint(0,len(self.QRS),size=1),self.N)
            id_ST = np.repeat(np.random.randint(0,len(self.ST),size=1),self.N)
            id_T = np.repeat(np.random.randint(0,len(self.T),size=1),self.N)
            id_TP = np.repeat(np.random.randint(0,len(self.TP),size=1),self.N)

        # In case QRS is not expressed
        filt_QRS = np.random.rand(self.N) > (1-self.proba_no_QRS)

        # P wave
        id_P[(np.random.rand(self.N) < self.proba_no_P) | does_not_have_P | has_TV] = -1
        id_PQ[filt_QRS | (np.random.rand(self.N) < self.proba_no_PQ) | does_not_have_PQ | has_TV] = -1
        id_QRS[filt_QRS] = -1
        id_ST[filt_QRS | (np.random.rand(self.N) < self.proba_no_ST) | does_not_have_ST] = -1
        id_T[filt_QRS] = -1
        id_TP[np.full((self.N),has_TV,dtype=bool)] = -1
        
        beats = []
        masks = []
        ids = []
        offset = 0
        record_size = 0
        mark_break = False
        for i in range(self.N):
            for j in range(6):
                if (i == 0) and (j < begining_wave): 
                    continue
                if (j == 0) and (id_P[i] != -1):
                    amplitude = self.Pamplitudes.rvs(1)
                    segment = amplitude*self.P[self.Pkeys[id_P[i]]]
                    segment *= 0.15*np.random.randn(1)+1
                    if has_interpolation[len(beats)]:
                        x = np.linspace(0,1,segment.size)
                        x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.5)).clip(1)))
                        segment = interp1d(x,segment)(x_new)
                    if has_elevation[len(beats)]:
                        right_amplitude = amplitude*norm.rvs(0,0.5,1)
                        right_sign = np.sign(right_amplitude)
                        segment += right_sign*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                    segment = trailonset(segment,offset)
                    beats.append(segment)
                    masks.append(np.full((beats[-1].size,),1,dtype='int8'))
                    ids.append(('P',self.Pkeys[id_P[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 1) and (id_PQ[i] != -1):
                    amplitude = self.PQamplitudes.rvs(1)
                    segment = amplitude*self.PQ[self.PQkeys[id_PQ[i]]]
                    segment *= 0.15*np.random.randn(1)+1
                    if has_interpolation[len(beats)]:
                        x = np.linspace(0,1,segment.size)
                        x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.5)).clip(1)))
                        segment = interp1d(x,segment)(x_new)
                    if has_elevation[len(beats)]:
                        right_amplitude = amplitude*norm.rvs(0,1.0,1)
                        right_sign = np.sign(right_amplitude)
                        segment += right_sign*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                    segment = trailonset(segment,offset)
                    beats.append(segment)
                    masks.append(np.zeros((beats[-1].size,),dtype='int8'))
                    ids.append(('PQ',self.PQkeys[id_PQ[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 2) and (id_QRS[i] != -1):
                    amplitude = self.QRSamplitudes.rvs(1)
                    segment = amplitude*self.QRS[self.QRSkeys[id_QRS[i]]]
                    segment *= 0.15*np.random.randn(1)+1
                    if has_interpolation[len(beats)]:
                        x = np.linspace(0,1,segment.size)
                        x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.5)).clip(1)))
                        segment = interp1d(x,segment)(x_new)
                    if has_elevation[len(beats)]:
                        right_amplitude = amplitude*norm.rvs(0,0.1,1)
                        right_sign = np.sign(right_amplitude)
                        segment += right_sign*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                    segment = trailonset(segment,offset)
                    beats.append(segment)
                    masks.append(np.full((beats[-1].size,),2,dtype='int8'))
                    ids.append(('QRS',self.QRSkeys[id_QRS[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 3) and (id_ST[i] != -1):
                    if has_TV:
                        nx = np.random.randint(16)
                        if nx < 2:
                            continue
                        stsegment = np.convolve(np.cumsum(norm.rvs(scale=0.01**(2*0.5),size=nx)),np.hamming(nx)/(nx//2),mode='same')
                    else:
                        stsegment = self.ST[self.STkeys[id_ST[i]]]
                    amplitude = self.STamplitudes.rvs(1)
                    segment = amplitude*stsegment
                    segment *= 0.15*np.random.randn(1)+1
                    if has_interpolation[len(beats)]:
                        x = np.linspace(0,1,segment.size)
                        x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.5)).clip(1)))
                        segment = interp1d(x,segment)(x_new)
                    if has_elevation[len(beats)]:
                        right_amplitude = amplitude*norm.rvs(0,1.5,1)
                        right_sign = np.sign(right_amplitude)
                        segment += right_sign*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                    segment = trailonset(segment,offset)
                    beats.append(segment)
                    masks.append(np.zeros((beats[-1].size,),dtype='int8'))
                    ids.append(('ST',self.STkeys[id_ST[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 4) and (id_T[i] != -1):
                    amplitude = self.Tamplitudes.rvs(1)
                    segment = amplitude*self.T[self.Tkeys[id_T[i]]]
                    segment *= 0.15*np.random.randn(1)+1
                    if has_interpolation[len(beats)]:
                        x = np.linspace(0,1,segment.size)
                        x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.5)).clip(1)))
                        segment = interp1d(x,segment)(x_new)
                    if has_elevation[len(beats)]:
                        right_amplitude = amplitude*norm.rvs(0,0.25,1)
                        right_sign = np.sign(right_amplitude)
                        segment += right_sign*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                    segment = trailonset(segment,offset)
                    beats.append(segment)
                    masks.append(np.full((beats[-1].size,),3,dtype='int8'))
                    ids.append(('T',self.Tkeys[id_T[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 5) and (id_TP[i] != -1):
                    amplitude = self.TPamplitudes.rvs(1)
                    segment = amplitude*self.TP[self.TPkeys[id_TP[i]]]
                    segment *= 0.15*np.random.randn(1)+1
                    if has_interpolation[len(beats)]:
                        x = np.linspace(0,1,segment.size)
                        x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.5)).clip(1)))
                        segment = interp1d(x,segment)(x_new)
                    if has_elevation[len(beats)]:
                        right_amplitude = amplitude*norm.rvs(0,0.15,1)
                        right_sign = np.sign(right_amplitude)
                        segment += right_sign*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                    segment = trailonset(segment,offset)
                    beats.append(segment)
                    masks.append(np.zeros((beats[-1].size,),dtype='int8'))
                    ids.append(('TP',self.TPkeys[id_TP[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (math.floor(record_size*interp_length)-onset) >= self.N:
                    mark_break = True
                    break
            if mark_break:
                break

        # Obtain final stuff
        signal = np.concatenate(beats)
        masks = np.concatenate(masks)

        # Interpolate signal & mask
        x = np.linspace(0,1,signal.size)
        signal = interp1d(x,signal)(np.linspace(0,1,math.ceil(signal.size*interp_length)))
        masks = interp1d(x,masks)(np.linspace(0,1,math.ceil(masks.size*interp_length))).astype(int)

        # Move onset
        signal = signal[onset:onset+self.N]
        masks = masks[onset:onset+self.N]

        # Modify amplitude
        signal = signal*global_amplitude

        # Express as masks
        if self.labels_as_masks:
            masks_all = np.zeros((3,self.N),dtype=bool)
            masks_all[0,:] = (masks == 1)
            masks_all[1,:] = (masks == 2)
            masks_all[2,:] = (masks == 3)
        else:
            masks_all = masks.astype('int32')

        # Add baseline wander
        if self.add_baseline_wander:
            signal = signal + np.convolve(np.cumsum(norm.rvs(scale=0.01**(2*0.5),size=self.N)),np.hamming(self.window)/(self.window/2),mode='same')
        
        return signal[None,].astype('float32'), masks_all
