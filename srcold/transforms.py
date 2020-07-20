import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt as dist_transform

def DistanceMapTransform(DataStorage, mode='log'):
    assert mode.lower() in ('log', 'linear')

    # For every element in the database
    for k in DataStorage.P.wave.keys():
        # Store distances
        DataStorage.P.wave[k]   = GetDistanceMap(DataStorage.P.wave[k], mode)
        DataStorage.QRS.wave[k] = GetDistanceMap(DataStorage.QRS.wave[k], mode)
        DataStorage.T.wave[k]   = GetDistanceMap(DataStorage.T.wave[k], mode)


def GetDistanceMap(mask, mode='log'):
    assert mode.lower() in ('log', 'linear')

    # Create storage for distance map
    distance = np.zeros_like(mask)

    # Due to edge calculation method in ContourLoss, masks have to extend 1 sample to the left
    mask = np.logical_or(np.pad(mask,((0,1),), 'edge')[1:],mask).values

    # Obtain the distance transform
    distance = dist_transform(np.logical_not(mask))*np.logical_not(mask) - (dist_transform(mask)-1)*mask

    # Clip and log it
    if mode.lower() == 'log': distance = np.log(distance.clip(min=0.5)) - np.log(-distance.clip(max=-0.5))

    return distance


def SignalPower(X):
    return np.mean((X - np.median(X))**2)

def additive_white_gaussian_noise(X, snr):
    # Compute signal power
    Power               = SignalPower(X)

    # Calculate signal to noise ratio with randomness
    SignalToNoiseRatio  = snr + np.random.uniform(low=-snr/10, high=snr/10)
    NoisePower          = Power/10**(SignalToNoiseRatio/10.)
    Noise               = np.random.normal(0,np.sqrt(NoisePower),len(X))

    return Noise

def random_spikes(X, snr, period):
    # Compute signal power
    Power               = SignalPower(X)

    # Specify number of samples
    N = np.random.randint(7,13)

    # Define a randomly initialized filter bank
    F = np.zeros((5,))
    F[0] = np.random.uniform(-0.15,0.25,1)[0]
    F[1] = np.random.uniform(0.25,0.5,1)[0]
    F[2] = np.random.uniform(1,2,1)[0]
    F[3] = np.random.uniform(-0.5,0.25,1)[0]
    F[4] = np.random.uniform(0,0.25,1)[0]

    # Interpolate to number of samples
    interp = interp1d(np.linspace(0,1,F.size), F, kind='quadratic')
    F = interp(np.linspace(0,1,N))
    E = (F**2).sum()
    F = F/np.sqrt(E)

    SNRdb         = snr + np.random.uniform(low=-snr/N, high=snr/N)
    T             = period + np.random.randint(low=-period/4, high=period/4)
    P             = np.random.randint(low=0,  high=T)
    
    # Train of deltas
    Noise         = np.zeros(X.shape)
    Noise[P::T]   = 1

    # Compute real period of the signal
    Treal         = Noise.size/Noise.sum()

    # Compute noise power
    NoisePower    = Power/10**(SNRdb/10.)
    Amplitude     = np.sqrt(NoisePower*Treal)

    # Specify filter normalized to power
    F             = Amplitude*F

    # Convolution of deltas
    Noise         = np.convolve(Noise, F, 'same')

    return Noise

def sinusoidal_noise(X, snr, freq):
    # Compute signal power
    Power               = SignalPower(X)

    SNRdb         = snr + np.random.uniform(low=-snr/10, high=snr/10)
    Freq          = freq + np.random.uniform(low=-freq/4, high=freq/4)

    NormFreq      = 2.*np.pi*Freq/250.
    NoisePower    = Power/10**(SNRdb/10.)
    Amplitude     = np.sqrt(2*NoisePower)
    Noise         = Amplitude*np.sin(NormFreq*np.arange(len(X)) + np.random.uniform(low=-np.pi, high=np.pi)) # Random initial phase

    return Noise

def pacemaker_spikes(X, y, snr):
    # Compute signal power
    Power               = SignalPower(X)

    SNRdb         = snr + np.random.uniform(low=-snr/10, high=snr/10)

    NoisePower    = Power/10**(SNRdb/10.)
    Onset         = np.where(np.diff(y) == 1)[0]
    Onset         = Onset + np.random.randint(low=-3,high=1, size=Onset.size) # 1 is never included
    Onset         = Onset[Onset >= 0]
    Amplitude     = np.sqrt(NoisePower*int(float(len(X))/len(Onset)))
    Noise         = np.zeros(X.shape)
    Noise[Onset]  = Amplitude

    return Noise

def amplifier_saturation(X, threshold):
    Noise         = np.zeros(X.shape)
    Maxmin        = np.max(np.abs(X))
    Percentile    = threshold + np.random.uniform(low=-threshold/4, high=threshold/4) # Random percentile change

    # Saturate on a percentage of the maximum value
    SatValue      = np.abs(Maxmin*(100-Percentile)/100)

    # Saturate on range (-oo, -v] U [v, +oo), as saturation is symmetrical
    Noise[X >=  SatValue] =  SatValue - X[X >=  SatValue]
    Noise[X <= -SatValue] = -SatValue - X[X <= -SatValue]

    return Noise

def DataAugmentationTransform(X, label, parameters, y):
    if (label   == 'awgn'):
        Noise = additive_white_gaussian_noise(X, parameters)
    elif (label == 'spikes'):
        Noise = random_spikes(X, *parameters)
    elif (label == 'powerline'):
        Noise = sinusoidal_noise(X, *parameters)
    elif (label == 'baseline'):
        Noise = sinusoidal_noise(X, parameters[0], 0)
    elif (label == 'pacemaker'):
        Noise = pacemaker_spikes(X, y, parameters)
    elif (label == 'sat_threshold'):
        Noise = amplifier_saturation(X, parameters)

    return Noise
