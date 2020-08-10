import torch
import torch.utils
import numpy as np
import math
from scipy.stats import norm
from scipy.interpolate import interp1d


def mixup(x1: np.ndarray, x2: np.ndarray, alpha: float = 1.0, beta: float = 1.0, axis = None, shuffle: bool = True):
    """Adapted from original authors of paper "[1710.09412] mixup: Beyond Empirical Risk Minimization"
    GitHub: https://github.com/facebookresearch/mixup-cifar10/
    """

    # Compute lambda. If hyperparams are incorrect, your loss
    lmbda = np.random.beta(alpha, beta)

    if axis is None:
        axis = 0
        shuffle = False # The default indicates that no batch is used

    # Swap axes to generalize for n-dimensional tensor
    x1 = np.swapaxes(x1,axis,0) # Compatible with pytorch
    x2 = np.swapaxes(x2,axis,0) # Compatible with pytorch

    # Permutation along data axis (allowing batch mixup)
    if shuffle:
        index = np.random.permutation(np.arange(x2.shape[0])) # Compatible with pytorch

        # Mix datapoints. If shapes are incompatible, your loss
        xhat = lmbda * x1 + (1 - lmbda) * x2[index, :]
    else:
        # Mix datapoints. If shapes are incompatible, your loss
        xhat = lmbda * x1 + (1 - lmbda) * x2
    
    # Swap axes back
    xhat = np.swapaxes(xhat,axis,0) # Compatible with pytorch

    # Return mixed point and lambda. Left label computation to be project-specific
    return xhat, lmbda


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
                 proba_mixup = 0.2, mixup_alpha = 1.0, mixup_beta = 1.0,
                 proba_TV = 0.05, proba_AF = 0.05, add_baseline_wander = True, 
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
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta

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

        # Utility
        self.eps = np.finfo('float').eps
        
        
    def __len__(self):
        '''Denotes the number of elements in the dataset'''
        return self.length

    def __getitem__(self, i: int):
        '''Generates one datapoint''' 
        # Set hyperparameters
        onset = np.random.randint(0,50)
        begining_wave = np.random.randint(0,6)
        global_amplitude = 1.+(np.random.randn(1)*self.amplitude_std)
        interp_length = max([1.+(np.random.randn(1)*self.interp_std),0.5])

        # Probabilities of waves
        does_not_have_P = (np.random.rand(1) > (1-self.proba_no_P))
        does_not_have_PQ = (np.random.rand(1) > (1-self.proba_no_PQ))
        does_not_have_ST = (np.random.rand(1) > (1-self.proba_no_ST))
        has_TV = (np.random.rand(1) > (1-self.proba_TV))
        has_AF = (np.random.rand(1) > (1-self.proba_AF))
        has_same_morph = (np.random.rand(1) > (1-self.proba_same_morph))
        has_elevation = (np.random.rand(self.N) > (1-self.proba_elevation))
        has_interpolation = (np.random.rand(self.N) > (1-self.proba_interpolation))
        has_mixup = (np.random.rand(self.N) > (1-self.proba_mixup))

        ##### Identifiers
        if has_same_morph:
            id_P = np.repeat(np.random.randint(0,len(self.P),size=1),self.N)
            id_PQ = np.repeat(np.random.randint(0,len(self.PQ),size=1),self.N)
            id_QRS = np.repeat(np.random.randint(0,len(self.QRS),size=1),self.N)
            id_ST = np.repeat(np.random.randint(0,len(self.ST),size=1),self.N)
            id_T = np.repeat(np.random.randint(0,len(self.T),size=1),self.N)
            id_TP = np.repeat(np.random.randint(0,len(self.TP),size=1),self.N)
        else:
            id_P = np.random.randint(0,len(self.P),size=self.N)
            id_PQ = np.random.randint(0,len(self.PQ),size=self.N)
            id_QRS = np.random.randint(0,len(self.QRS),size=self.N)
            id_ST = np.random.randint(0,len(self.ST),size=self.N)
            id_T = np.random.randint(0,len(self.T),size=self.N)
            id_TP = np.random.randint(0,len(self.TP),size=self.N)

        if has_AF:
            id_P = np.repeat(np.random.randint(0,len(self.P),size=1),self.N)
            pAF = self.P[self.Pkeys[id_P[0]]]
            pAF = interp1d(np.linspace(0,1,pAF.size),pAF)(np.linspace(0,1,pAF.size//2))
            pdist = 4*self.Pamplitudes.rvs(1)
            
        # In case QRS is not expressed
        filt_QRS = np.random.rand(self.N) > (1-self.proba_no_QRS)

        # P wave
        id_P[(np.random.rand(self.N) < self.proba_no_P) | does_not_have_P | has_TV] = -1
        id_PQ[filt_QRS | (np.random.rand(self.N) < self.proba_no_PQ) | does_not_have_PQ | has_TV] = -1
        id_QRS[filt_QRS] = -1
        id_ST[filt_QRS | (np.random.rand(self.N) < self.proba_no_ST) | does_not_have_ST] = -1
        id_T[filt_QRS] = -1
        id_TP[np.full((self.N),has_TV,dtype=bool)] = -1
        
        ##### Output data structure
        beats = []
        masks = []
        record_size = 0
        mark_break = False
        for i in range(self.N): # Unrealistic upper limit
            for j in range(6):
                if (i == 0) and (j < begining_wave): continue
                if j == 0: id,keys,waves,distribution = id_P[i],   self.Pkeys,   self.P,   self.Pamplitudes
                if j == 1: id,keys,waves,distribution = id_PQ[i],  self.PQkeys,  self.PQ,  self.PQamplitudes
                if j == 2: id,keys,waves,distribution = id_QRS[i], self.QRSkeys, self.QRS, self.QRSamplitudes
                if j == 3: id,keys,waves,distribution = id_ST[i],  self.STkeys,  self.ST,  self.STamplitudes
                if j == 4: id,keys,waves,distribution = id_T[i],   self.Tkeys,   self.T,   self.Tamplitudes
                if j == 5: id,keys,waves,distribution = id_TP[i],  self.TPkeys,  self.TP,  self.TPamplitudes
                if (math.floor(record_size*interp_length)-onset) >= self.N: 
                    mark_break = True
                    break
                
                # 0. Skip beats if not wanted, according to above rules
                if id == -1: continue # Flag for skipping unwanted beat
                
                # 1. Retrieve segment for modulation
                segment = waves[keys[id]].copy()

                # 1.1. If selected, apply mixup to segment
                if has_mixup[len(beats)]:
                    rand_id  = np.random.randint(0,len(keys))
                    segment2 = waves[keys[rand_id]]
                    if segment.size != segment2.size:
                        intlen   = np.random.randint(min([segment.size,segment2.size]),max([segment.size,segment2.size]))
                        segment  = interp1d(np.linspace(0,1,segment.size),segment)(np.linspace(0,1,intlen))
                        segment2 = interp1d(np.linspace(0,1,segment2.size),segment2)(np.linspace(0,1,intlen))
                    (segment,_) = mixup(segment,segment2,self.mixup_alpha,self.mixup_beta)
                    segment /= np.max(segment)-np.min(segment)+self.eps

                # 1.2. If selected, substitute ST segment by random walk
                if (j == 3) and has_TV:
                    nx = np.random.randint(32)
                    if nx < 2: continue # Avoid always having a TV with some space between QRS and T
                    segment = np.convolve(np.cumsum(norm.rvs(scale=0.01**(2*0.5),size=nx)),np.hamming(nx)/(nx//2),mode='same')

                # 1.2. If selected, convolve TP segment by P wave for AF simulation
                if has_AF:
                    if j == 0:
                        segment  = waves[keys[id]].copy()
                    if j == 5:
                        i = np.random.choice([0,1])
                        segment  = waves[keys[id]].copy()
                        segment  = np.convolve(segment,np.concatenate([((-1)**(i))*pAF,((-1)**(i+1))*pAF]))
                        segment /= np.max(segment)-np.min(segment)+self.eps

                # 2. Apply amplitude modulation
                # 2.1. Case has AF
                if has_AF and (j in [0,5]):
                    amplitude = pdist
                else:
                    amplitude = distribution.rvs(1)

                # 2.N. Apply the amplitude modulation on the segment
                segment *= amplitude
                
                # 3. Apply segment-wise augmentations.
                # 3.1. Per-segment amplitude noising
                noise = 0.15*np.random.randn(1)+1
                segment *= noise
                
                # 3.2. Per-segment interpolation
                if has_interpolation[len(beats)]:
                    x = np.linspace(0,1,segment.size)
                    x_new = np.linspace(0,1,int((segment.size*norm.rvs(1,0.25).clip(min=0.5))))
                    segment = interp1d(x,segment)(x_new)
                
                # 3.3. Per-segment right extrema elevation/depression
                if has_elevation[len(beats)]:
                    right_amplitude = distribution.rvs(1)*0.05
                    segment += np.random.choice([-1,1])*(np.linspace(0,np.sqrt(np.abs(right_amplitude)),segment.size)**2).squeeze()
                
                # 4. Segment-wise post-operations
                # 4.1. Onset trailing for coherency
                segment = trailonset(segment,beats[-1][-1] if len(beats) != 0 else 0)

                # 4.1. Final segment storage
                mask_value = 1 if j==0 else 2 if j==2 else 3 if j==4 else 0
                beats.append(segment)
                masks.append(np.full((segment.size,),mask_value,dtype='int8'))
                record_size += segment.size
            if mark_break:
                break

        # 5. Registry-wise post-operations
        # 5.1. Concatenate signal and mask
        signal = np.concatenate(beats)
        masks = np.concatenate(masks)

        # 5.2. Interpolate signal & mask
        x = np.linspace(0,1,signal.size)
        signal = interp1d(x,signal)(np.linspace(0,1,math.ceil(signal.size*interp_length)))
        masks = interp1d(x,masks)(np.linspace(0,1,math.ceil(masks.size*interp_length))).astype(int)

        # 5.3. Apply random onset for avoiding always starting with the same wave at the same location
        signal = signal[onset:onset+self.N]
        masks = masks[onset:onset+self.N]

        # 5.4. Apply global amplitude modulation
        signal = signal*global_amplitude

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
            signal = signal + np.convolve(np.cumsum(norm.rvs(scale=0.01**(2*0.5),size=self.N)),np.hamming(self.window)/(self.window/2),mode='same')
        
        return signal[None,].astype('float32'), masks_all, beats
