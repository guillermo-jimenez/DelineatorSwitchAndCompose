import torch
import torch.utils
import numpy as np
from scipy.stats import norm


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
                 length,N = 2048, noise = 0.005, proba_P = 0.25,
                 proba_QRS = 0.01, proba_PQ = 0.15, 
                 proba_ST = 0.15, proba_same_morph = 0.2,
                 add_baseline_wander = True, window = 51,
                 labels_as_masks = True):
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

        # Probabilities
        self.proba_P = proba_P
        self.proba_QRS = proba_QRS
        self.proba_PQ = proba_PQ
        self.proba_ST = proba_ST
        self.proba_same_morph = proba_same_morph
        
        
    def __len__(self):
        '''Denotes the number of elements in the dataset'''
        return self.length

    def __getitem__(self, i: int):
        '''Generates one datapoint''' 
        onset = np.random.randint(0,50)
        begining_wave = np.random.randint(0,6)
        has_P = (np.random.rand(1) > self.proba_P)
        has_PQ = (np.random.rand(1) > self.proba_PQ)
        has_ST = (np.random.rand(1) > self.proba_ST)
        has_same_morph = (np.random.rand(1) > self.proba_same_morph)

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
        filt_QRS = np.random.rand(self.N) < self.proba_QRS

        # P wave
        id_P[(np.random.rand(self.N) < self.proba_P) | np.logical_not(has_P)] = -1
        id_PQ[filt_QRS | (np.random.rand(self.N) < self.proba_PQ) | np.logical_not(has_PQ)] = -1
        id_QRS[filt_QRS] = -1
        id_ST[filt_QRS | (np.random.rand(self.N) < self.proba_ST) | np.logical_not(has_ST)] = -1
        id_T[filt_QRS] = -1

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
                    segment = trailonset(amplitude*self.P[self.Pkeys[id_P[i]]],offset)
                    segment *= 0.15*np.random.randn(1)+1
                    beats.append(segment)
                    masks.append(np.full((beats[-1].size,),1,dtype='int8'))
                    ids.append(('P',self.Pkeys[id_P[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 1) and (id_PQ[i] != -1):
                    amplitude = self.PQamplitudes.rvs(1)
                    segment = trailonset(amplitude*self.PQ[self.PQkeys[id_PQ[i]]],offset)
                    segment *= 0.15*np.random.randn(1)+1
                    beats.append(segment)
                    masks.append(np.zeros((beats[-1].size,),dtype='int8'))
                    ids.append(('PQ',self.PQkeys[id_PQ[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 2) and (id_QRS[i] != -1):
                    amplitude = self.QRSamplitudes.rvs(1)
                    segment = trailonset(amplitude*self.QRS[self.QRSkeys[id_QRS[i]]],offset)
                    segment *= 0.15*np.random.randn(1)+1
                    beats.append(segment)
                    masks.append(np.full((beats[-1].size,),2,dtype='int8'))
                    ids.append(('QRS',self.QRSkeys[id_QRS[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 3) and (id_ST[i] != -1):
                    amplitude = self.STamplitudes.rvs(1)
                    segment = trailonset(amplitude*self.ST[self.STkeys[id_ST[i]]],offset)
                    segment *= 0.15*np.random.randn(1)+1
                    beats.append(segment)
                    masks.append(np.zeros((beats[-1].size,),dtype='int8'))
                    ids.append(('ST',self.STkeys[id_ST[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 4) and (id_T[i] != -1):
                    amplitude = self.Tamplitudes.rvs(1)
                    segment = trailonset(amplitude*self.T[self.Tkeys[id_T[i]]],offset)
                    segment *= 0.15*np.random.randn(1)+1
                    beats.append(segment)
                    masks.append(np.full((beats[-1].size,),3,dtype='int8'))
                    ids.append(('T',self.Tkeys[id_T[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (j == 5) and (id_TP[i] != -1):
                    amplitude = self.TPamplitudes.rvs(1)
                    segment = trailonset(amplitude*self.TP[self.TPkeys[id_TP[i]]],offset)
                    segment *= 0.15*np.random.randn(1)+1
                    beats.append(segment)
                    masks.append(np.zeros((beats[-1].size,),dtype='int8'))
                    ids.append(('TP',self.TPkeys[id_TP[i]]))
                    offset = beats[-1][-1]
                    record_size += beats[-1].size
                if (record_size-onset) >= self.N:
                    mark_break = True
                    break
            if mark_break:
                break

        # Obtain final stuff
        signal = np.concatenate(beats,)
        masks = np.concatenate(masks)

        # Move onset
        signal = signal[onset:onset+self.N]
        masks = masks[onset:onset+self.N]

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
