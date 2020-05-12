""" Wave generator from fiducial data. Fast script for generating masks from re-annotated data"""

from utils.data_structures import load_data
from os.path import join
import pandas
import glob
import numpy as np

path = '/media/guille/DADES/DADES/PhysioNet/QTDB/manual0'

maxSize = 225000

################ P wave ################
on   = load_data(join(path, 'Pon.csv'))
off  = load_data(join(path, 'Poff.csv'))
wave = pandas.DataFrame()

# generate wave
for k in on.keys():
    wave[k] = np.zeros((maxSize,),dtype=int)

    for i in range(len(on[k])):
        wave[k][on[k][i]:off[k][i]] = 1

# Save results
wave.to_csv(join(path, 'Pwave.csv'))


############### QRS wave ###############
on   = load_data(join(path, 'QRSon.csv'))
off  = load_data(join(path, 'QRSoff.csv'))
wave = pandas.DataFrame()

# generate wave
for k in on.keys():
    wave[k] = np.zeros((maxSize,),dtype=int)

    for i in range(len(on[k])):
        wave[k][on[k][i]:off[k][i]] = 1

# Save results
wave.to_csv(join(path, 'QRSwave.csv'))


################ T wave ################
on   = load_data(join(path, 'Ton.csv'))
off  = load_data(join(path, 'Toff.csv'))
wave = pandas.DataFrame()

# generate wave
for k in on.keys():
    wave[k] = np.zeros((maxSize,),dtype=int)

    for i in range(len(on[k])):
        wave[k][on[k][i]:off[k][i]] = 1

# Save results
wave.to_csv(join(path, 'Twave.csv'))

