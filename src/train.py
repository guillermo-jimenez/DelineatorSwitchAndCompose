import numpy as np
import torch
import sak
import sak.torch

def get_batch_ssl(loader_labeled,loader_unlabeled):
    # Set up all stuff
    counter_labeled,counter_unlabeled =                    0,                      0
    length_labeled,length_unlabeled   =  len(loader_labeled),  len(loader_unlabeled)
    iter_labeled,    iter_unlabeled   = iter(loader_labeled), iter(loader_unlabeled)
    maxlen                            =  max(length_labeled,       length_unlabeled)
    
    # Get examples
    for _ in range(maxlen):
        if counter_labeled   == length_labeled:
            iter_labeled   = iter(loader_labeled)
        if counter_unlabeled == length_unlabeled:
            iter_unlabeled = iter(loader_unlabeled)
        counter_labeled   += 1
        counter_unlabeled += 1

        # Return next element in iterator
        yield (next(iter_labeled), next(iter_unlabeled))

