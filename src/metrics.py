from typing import List
import numpy as np


def dice_score(input: np.ndarray, target: np.ndarray) -> float:
    intersection = (input * target).sum()
    union = input.sum() + target.sum()
    return 2.*intersection/(union + np.finfo('double').eps)


def filter_valid(onset: int, offset: int, validity_on: int = 0, validity_off: int = np.inf):
    validity_on  = np.array( validity_on)[np.newaxis,np.newaxis]
    validity_off = np.array(validity_off)[np.newaxis,np.newaxis]

    mask_on    = (onset  >= validity_on) & (onset  <= validity_off)
    mask_off   = (offset >= validity_on) & (offset <= validity_off)
    mask_total = np.any(mask_on & mask_off, axis=0) # beat has to be found in every one

    onset = onset[mask_total]
    offset = offset[mask_total]

    return onset, offset


def correspondence(input_onsets, input_offsets, target_onsets, target_offsets):
    filtA =  ( input_onsets <=  target_onsets[:,np.newaxis]) & ( target_onsets[:,np.newaxis] <= input_offsets)
    filtB =  ( input_onsets <= target_offsets[:,np.newaxis]) & (target_offsets[:,np.newaxis] <= input_offsets)
    filtC = ((target_onsets <=   input_onsets[:,np.newaxis]) & (  input_onsets[:,np.newaxis] <= target_offsets)).T
    filtD = ((target_onsets <=  input_offsets[:,np.newaxis]) & ( input_offsets[:,np.newaxis] <= target_offsets)).T

    filter = filtA | filtB | filtC | filtD

    return filter


# def interlead_correspondence(input_onsets_list: List[np.ndarray], input_offsets_list: List[np.ndarray], 
#                              target_onsets_list: List[np.ndarray], target_offsets_list: List[np.ndarray], 
#                              validity_on: int, validity_off: int):
#     # ##### NOT FINISHED #####
#     # filtA =  (res_0_on <= res_1_on[:,np.newaxis]) & (res_1_on[:,np.newaxis] <= res_0_of)
#     # filtB =  (res_0_on <= res_1_of[:,np.newaxis]) & (res_1_of[:,np.newaxis] <= res_0_of)
#     # filtC = ((res_1_on <= res_0_on[:,np.newaxis]) & (res_0_on[:,np.newaxis] <= res_1_of)).T
#     # filtD = ((res_1_on <= res_0_of[:,np.newaxis]) & (res_0_of[:,np.newaxis] <= res_1_of)).T
#     # filter = filtA | filtB | filtC | filtD
#     # return filter
#     pass


def post_processing(input_onsets: np.ndarray,input_offsets: np.ndarray,
                    target_onsets: np.ndarray,target_offsets: np.ndarray,
                    validity_on: int,validity_off: int):
    input_onsets,input_offsets = filter_valid(input_onsets,input_offsets,validity_on,validity_off)
    target_onsets,target_offsets = filter_valid(target_onsets,target_offsets,validity_on,validity_off)
    
    return input_onsets,input_offsets,target_onsets,target_offsets


def compute_metrics(input_onsets: np.ndarray, input_offsets: np.ndarray, 
                    target_onsets: np.ndarray, target_offsets: np.ndarray):
    # Init output
    tp   = 0
    fp   = 0
    fn   = 0
    dice = 0
    onset_error  = []
    offset_error = []

    # Find correspondence between fiducials
    filter = correspondence(input_onsets, input_offsets, target_onsets, target_offsets)

    # Check correspondence of GT beats to detected beats
    corr  = dict()
    
    # Account for already detected beats to calculate false positives
    chosen = np.zeros((filter.shape[0],), dtype=bool)
    for i,column in enumerate(filter.T):
        corr[i] = np.where(column)[0]
        chosen = chosen | column
        
    # Retrieve beats detected that do not correspond to any GT beat (potential false positives)
    not_chosen = np.where(np.logical_not(chosen))[0]
    
    # Compute Dice coefficient
    mask_input  = np.zeros((np.max(np.hstack((input_offsets,target_offsets)))+10,),dtype=bool)
    mask_target = np.zeros((np.max(np.hstack((input_offsets,target_offsets)))+10,),dtype=bool)
    for (onset,offset) in zip(input_onsets,input_offsets):
        mask_input[onset:offset] = True
    for (onset,offset) in zip(target_onsets,target_offsets):
        mask_target[onset:offset] = True
    dice = dice_score(mask_input, mask_target)

    # Compute metrics - Fusion strategy of results of both leads, following Martinez et al.
    for i in range(filter.shape[1]):
        # If any GT beat has a correspondence to any segmented beat, true positive + accounts for on/offset error
        if len(corr[i]) != 0:
            # Mark beat as true positive
            tp += 1
            
            # Compute the onset-offset errors
            onset_error.append(int(target_onsets[corr[i]]  - input_onsets[i]))
            offset_error.append(int(target_offsets[corr[i]] - input_offsets[i]))
            
        # If any GT beat has a correspondence to more than one segmented beat, 
        #     the rest of the pairs have to be false positives (Martinez et al.)
        if len(corr[i]) > 1:
            fp += len(corr[i]) - 1
        
        # If any GT beat has no correspondence to any segmented beat, false negative
        if len(corr[i]) == 0:
            fn += 1
            
    # False positives will correspond to those existing in the results that do not correspond to any beat in the GT (the not chosen)
    fp += len(not_chosen)
    
    return tp,fp,fn,dice,onset_error,offset_error
        

def precision(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fp)

def recall(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fn)

def f1_score(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+(fp+fn)/2)

