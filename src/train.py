from typing import Any, Tuple, List, Callable

import os
import csv
import time
import copy
import dill
import os.path
import shutil
import random
import tqdm
import torch
import torch.nn
import numpy as np
import sak
import sak.torch
import sak.torch.data

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


def do_epoch_ssl(model_student: torch.nn.Module, model_teacher: torch.nn.Module, state: dict, config: dict, 
                 dataloader_labeled:   torch.utils.data.DataLoader, 
                 dataloader_unlabeled: torch.utils.data.DataLoader, 
                 criterion: Callable, 
                 criterion_crossentropy: Callable) -> list:
    # Log progress
    max_length = max(len(dataloader_labeled),len(dataloader_unlabeled))
    batch_loss = np.zeros((max_length,2),dtype='float16')
    generator  = get_batch_ssl(dataloader_labeled,dataloader_unlabeled)
    state["moving_dot_product"] = state.get("moving_dot_product",0)

    # Create transforms
    if ('data_pre' in config):
        data_pre     = sak.from_dict(config["data_pre"])
    if ('augmentation' in config) and model_student.training:
        augmentation = sak.from_dict(config["augmentation"])
    if ('data_post' in config):
        data_post    = sak.from_dict(config["data_post"])

    # Select iterator decorator
    train_type = 'Train' if model_student.training else 'Valid'
    iterator = sak.get_tqdm(generator, config.get('iterator',''), 
                            desc="({}) Epoch {:>3d}/{:>3d}, Loss {:0.3f}".format(train_type, 
                                                                                state['epoch']+1, 
                                                                                config['epochs'], np.inf))

    for i,(inputs_sup,inputs_unsup) in enumerate(iterator):
        # Apply data transforms
        if ('data_pre' in config):
            data_pre(inputs=inputs_sup)
            data_pre(inputs=inputs_unsup)
        if ('augmentation' in config) and model_student.training:
            augmentation(inputs=inputs_sup)
            augmentation(inputs=inputs_unsup)
        if ('data_post' in config):
            data_post(inputs=inputs_sup)
            data_post(inputs=inputs_unsup)

        # UDA
        inputs_aug = copy.deepcopy(inputs_unsup)
        if ('augmentation' in config) and model_student.training:
            for _ in range(random.randint(0,config["UDA"].get("max_N",5)-2)):
                augmentation(inputs=inputs_aug)
                
        # Map models to device
        model_teacher = model_teacher.to(state['device'], non_blocking=True)
        model_student = model_student.to(state['device'], non_blocking=True)
        
        # Set gradients to zero
        if model_teacher.training: state['optimizer_teacher'].zero_grad()
        if model_student.training: state['optimizer_student'].zero_grad()
        
        # Predict supervised data
        inputs_sup            = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_sup.items()}
        outputs_sup_teacher   = {k: v.cpu() for k,v in model_teacher(inputs_sup).items()}
        outputs_sup_student   = {k: v.cpu() for k,v in model_student(inputs_sup).items()}
        inputs_sup            = {k: v.cpu() for k,v in inputs_sup.items()}
        
        # Predict unsupervised data
        inputs_unsup          = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_unsup.items()}
        outputs_unsup_teacher = {k: v.cpu() for k,v in model_teacher(inputs_unsup).items()}
        inputs_unsup          = {k: v.cpu() for k,v in inputs_unsup.items()}

        # Predict augmented data
        inputs_aug            = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_aug.items()}
        outputs_aug_teacher   = {k: v.cpu() for k,v in model_teacher(inputs_aug).items()}
        outputs_aug_student   = {k: v.cpu() for k,v in model_student(inputs_aug).items()}
        inputs_aug            = {k: v.cpu() for k,v in inputs_aug.items()}
        
        # Supervised loss UDA
        uda_sup_loss          = criterion(inputs=inputs_sup,outputs=outputs_sup_teacher,state=state)
        
        # Temperature on unsupervised for UDA
        outputs_unsup_teacher_temp = {k: v.clone().detach() for k,v in outputs_unsup_teacher.items()}
        outputs_unsup_teacher_temp["logits"] = outputs_unsup_teacher_temp["logits"]/config["UDA"].get("temperature",0.7)
        outputs_unsup_teacher_temp["probas"] = torch.sigmoid(outputs_unsup_teacher_temp["logits"])
        
        # Mask outputs for unsupervised loss
        largest_probs,_       = torch.max(outputs_unsup_teacher_temp["probas"].detach().flatten(start_dim=1),dim=-1)
        mask_elements         = torch.ge(largest_probs,config["UDA"].get("mask_threshold",0.6)).float() # GOOD ONE
        for _ in range(outputs_unsup_teacher_temp["probas"].ndim-1):
            mask_elements     = mask_elements.unsqueeze(-1)
        outputs_aug_teacher_mask   = {k: v.clone()*mask_elements for k,v in outputs_aug_teacher.items()}
        outputs_unsup_teacher_temp = {k: v.clone()*mask_elements for k,v in outputs_unsup_teacher_temp.items()}
        outputs_unsup_teacher_temp["y"] = outputs_unsup_teacher_temp["probas"]
            
        # Unsupervised loss UDA
        uda_unsup_loss        = criterion(inputs=outputs_unsup_teacher_temp,outputs=outputs_aug_teacher_mask,state=state)

        # Total loss UDA
        step                  = max_length*state["epoch"] + i
        uda_weight_decay      = config["UDA"].get("weight_decay",0) # compute weight decay
        uda_weight            = config["UDA"].get("weight",10)*min(step/config["UDA"].get("steps",1),1)
        uda_total_loss        = (uda_weight*uda_unsup_loss) + uda_sup_loss + uda_weight_decay
        

        #########################
        # Loss on UDA labels, "for backprop" and "For Taylor"
        outputs_aug_teacher_detached = {k: v.clone().detach() for k,v in outputs_aug_teacher.items()}
        outputs_aug_teacher_detached["y"] = outputs_aug_teacher_detached["probas"]
        student_loss          = criterion(inputs=outputs_aug_teacher_detached, outputs=outputs_aug_student,state=state)
        student_loss_cce_prev = criterion(inputs=inputs_sup, outputs=outputs_sup_student,state=state)

        # Optimize network's weights
        # Break early
        if torch.isnan(student_loss):
            continue
            # raise ValueError("Nan loss value encountered. Stopping...")
        if model_student.training:
            student_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_student.parameters(),config["UDA"].get("grad_clip",5.0))
            state["optimizer_student"].step()
            if "lr_scheduler_student" in state:
                state["lr_scheduler_student"].step()

        # Apply Exponential Moving Average of student model
        if config["UDA"].get("ema",0) > 0: 
            state["ema_model"].update_parameters(model_student)

        # 2nd call to student: get logits & cross-entropy
        # Predict supervised data
        inputs_sup               = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_sup.items()}
        outputs_sup_student_next = {k: v.cpu() for k,v in model_student(inputs_sup).items()}
        inputs_sup               = {k: v.cpu() for k,v in inputs_sup.items()}
        
        # Get supervised loss
        student_loss_cce_next    = criterion(inputs=inputs_sup, outputs=outputs_sup_student_next,state=state)

        # Get dot product, moving average of dot product & correct dot product with moving average
        dot_product              = (student_loss_cce_next - student_loss_cce_prev).detach()
        moving_dot_product       = state["moving_dot_product"] + 0.01*(dot_product-state["moving_dot_product"])
        dot_product             -= state["moving_dot_product"]

        # Final teacher loss
        crossentropy_loss        = criterion_crossentropy(inputs=outputs_aug_teacher_detached,outputs=outputs_aug_teacher,state=state)
        teacher_loss             = uda_total_loss + dot_product*crossentropy_loss

        # Optimize teacher
        if torch.isnan(teacher_loss):
            continue
            # raise ValueError("Nan loss value encountered. Stopping...")
        if model_teacher.training:
            teacher_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_teacher.parameters(),config["UDA"].get("grad_clip",5.0))
            state["optimizer_teacher"].step()
            if "lr_scheduler_teacher" in state:
                state["lr_scheduler_teacher"].step()
        
        # Retrieve for printing purposes
        print_loss = (student_loss.item(),teacher_loss.item())

        # Accumulate losses
        batch_loss[i] = print_loss
        
        # Change iterator description
        if isinstance(iterator,tqdm.tqdm):
            if i == max_length-1: batch_print = np.mean(batch_loss,axis=0)
            else:                 batch_print = print_loss
            iterator.set_description("({}) Epoch {:>3d}/{:>3d}, Loss {:10.3f}, {:10.3f}".format(train_type, state['epoch']+1, config['epochs'], batch_print[0], batch_print[1]))
            iterator.refresh()

    return batch_loss


def train_model_ssl(model_student: torch.nn.Module, model_teacher: torch.nn.Module, 
                    state: dict, config: dict, loader_train: torch.utils.data.DataLoader,
                    loader_unsupervised: torch.utils.data.DataLoader):
    # Send model to device
    model_student = model_student.to(state['device'])
    model_teacher = model_teacher.to(state['device'])

    # Instantiate criterion
    criterion              = sak.from_dict(config['loss'])
    criterion_crossentropy = sak.from_dict(config['loss_crossentropy'])
    
    # Initialize best loss for early stopping
    if 'best_loss' not in state:
        state['best_loss'] = [np.inf,np.inf]

    # Get savedir string
    if   "savedir"        in config: str_savedir = 'savedir'
    elif "save_directory" in config: str_savedir = 'save_directory'
    else: raise ValueError("Configuration file should include either the 'savedir' or 'save_directory' fields [case-sensitive]")

    # Iterate over epochs
    for epoch in range(state['epoch'], config['epochs']):
        try:
            # Store current epoch
            state['epoch'] = epoch
            
            # Train model
            loss_train          = do_epoch_ssl(model_student.train(), model_teacher.train(), state, config, 
                                               loader_train, loader_unsupervised, criterion, criterion_crossentropy)

            # Save model/state info
            model_student = model_student.cpu().eval()
            model_teacher = model_teacher.cpu().eval()
            torch.save(model_student,              os.path.join(config[str_savedir],'checkpoint_student.model'),      pickle_module=dill)
            torch.save(model_teacher,              os.path.join(config[str_savedir],'checkpoint_teacher.model'),      pickle_module=dill)
            torch.save(model_student.state_dict(), os.path.join(config[str_savedir],'checkpoint_student.state_dict'), pickle_module=dill)
            torch.save(model_teacher.state_dict(), os.path.join(config[str_savedir],'checkpoint_teacher.state_dict'), pickle_module=dill)
            sak.pickledump(state, os.path.join(config[str_savedir],'checkpoint.state'), mode='wb')
            model_student = model_student.to(state['device'])
            model_teacher = model_teacher.to(state['device'])
          
            # Log train loss
            with open(os.path.join(config[str_savedir],'log.csv'),'a') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(["(Train) Epoch {:>3d}/{:>3d}, Loss {:10.3f}, {:10.3f}, Time {}".format(state['epoch']+1, config['epochs'], state['loss_train'][0], state['loss_train'][1], time.ctime())])

            # Check if loss is best loss
            if state['loss_train'][0] < state['best_loss']:
                state['best_loss']  = state['loss_train'][0]
                state['best_epoch'] = epoch
                
                # Copy checkpoint and mark as best
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_student.model'), os.path.join(config[str_savedir],'model_best_student.model'))
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_teacher.model'), os.path.join(config[str_savedir],'model_best_teacher.model'))
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_student.state'), os.path.join(config[str_savedir],'model_best_student.state'))
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_teacher.state'), os.path.join(config[str_savedir],'model_best_teacher.state'))
        except KeyboardInterrupt:
            model_student = model_student.cpu().eval()
            model_teacher = model_teacher.cpu().eval()
            torch.save(model_student,              os.path.join(config[str_savedir],'keyboard_interrupt_student.model'),      pickle_module=dill)
            torch.save(model_teacher,              os.path.join(config[str_savedir],'keyboard_interrupt_teacher.model'),      pickle_module=dill)
            torch.save(model_student.state_dict(), os.path.join(config[str_savedir],'keyboard_interrupt_student.state_dict'), pickle_module=dill)
            torch.save(model_teacher.state_dict(), os.path.join(config[str_savedir],'keyboard_interrupt_teacher.state_dict'), pickle_module=dill)
            sak.pickledump(state, os.path.join(config[str_savedir],'keyboard_interrupt.state'), mode='wb')
            raise
        except:
            model_student = model_student.cpu().eval()
            model_teacher = model_teacher.cpu().eval()
            torch.save(model_student,              os.path.join(config[str_savedir],'error_student.model'),      pickle_module=dill)
            torch.save(model_teacher,              os.path.join(config[str_savedir],'error_teacher.model'),      pickle_module=dill)
            torch.save(model_student.state_dict(), os.path.join(config[str_savedir],'error_student.state_dict'), pickle_module=dill)
            torch.save(model_teacher.state_dict(), os.path.join(config[str_savedir],'error_teacher.state_dict'), pickle_module=dill)
            sak.pickledump(state, os.path.join(config[str_savedir],'error.state'), mode='wb')
            raise


def train_valid_model_ssl(model_student, model_teacher, state: dict, config: dict, 
                          loader_train: torch.utils.data.DataLoader, 
                          loader_unsupervised: torch.utils.data.DataLoader, 
                          loader_valid: torch.utils.data.DataLoader):
    # Send model to device
    model_student = model_student.to(state['device'])
    model_teacher = model_teacher.to(state['device'])

    # Instantiate criterion
    criterion              = sak.from_dict(config['loss'])
    criterion_crossentropy = sak.from_dict(config['loss_crossentropy'])
    
    # Initialize best loss for early stopping
    if 'best_loss' not in state:
        state['best_loss'] = [np.inf,np.inf]

    # Get savedir string
    if   "savedir"        in config: str_savedir = 'savedir'
    elif "save_directory" in config: str_savedir = 'save_directory'
    else: raise ValueError("Configuration file should include either the 'savedir' or 'save_directory' fields [case-sensitive]")

    # Iterate over epochs
    for epoch in range(state['epoch'], config['epochs']):
        try:
            # Store current epoch
            state['epoch'] = epoch
            
            # Train model
            loss_train     = do_epoch_ssl(model_student.train(), model_teacher.train(), state, config, 
                                          loader_train, loader_unsupervised, criterion, criterion_crossentropy)

            # Validate results
            with torch.no_grad():
                loss_valid = sak.torch.train.do_epoch(model_student.eval(), state, config, loader_valid, criterion)
            state['loss_validation'] = np.mean(loss_valid)

            # Update learning rate scheduler
            if 'scheduler' in state:
                state['scheduler'].step(state['loss_validation'])

            # Save model/state info
            model_student = model_student.cpu().eval()
            model_teacher = model_teacher.cpu().eval()
            torch.save(model_student,              os.path.join(config[str_savedir],'checkpoint_student.model'),      pickle_module=dill)
            torch.save(model_teacher,              os.path.join(config[str_savedir],'checkpoint_teacher.model'),      pickle_module=dill)
            torch.save(model_student.state_dict(), os.path.join(config[str_savedir],'checkpoint_student.state_dict'), pickle_module=dill)
            torch.save(model_teacher.state_dict(), os.path.join(config[str_savedir],'checkpoint_teacher.state_dict'), pickle_module=dill)
            sak.pickledump(state, os.path.join(config[str_savedir],'checkpoint.state'), mode='wb')
            model_student = model_student.to(state['device'])
            model_teacher = model_teacher.to(state['device'])
            
            # Log train loss
            with open(os.path.join(config[str_savedir],'log.csv'),'a') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(["(Train) Epoch {:>3d}/{:>3d}, Loss {:10.3f}, {:10.3f}, Time {}".format(state['epoch']+1, config['epochs'], state['loss_train'][0], state['loss_train'][1], time.ctime())])
                csvwriter.writerow(["(Valid) Epoch {:>3d}/{:>3d}, Loss {:10.3f}, {:10.3f}, Time {}".format(state['epoch']+1, config['epochs'], state['loss_validation'], 0.0, time.ctime())])

            # Check if loss is best loss
            compound_loss = 2*state['loss_train'][0]*state['loss_validation']/(state['loss_train'][0]+state['loss_validation'])
            if compound_loss < state['best_loss']:
                state['best_loss']  = compound_loss
                state['best_epoch'] = epoch
                
                # Copy checkpoint and mark as best
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_student.model'), os.path.join(config[str_savedir],'model_best_student.model'))
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_teacher.model'), os.path.join(config[str_savedir],'model_best_teacher.model'))
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_student.state'), os.path.join(config[str_savedir],'model_best_student.state'))
                shutil.copyfile(os.path.join(config[str_savedir],'checkpoint_teacher.state'), os.path.join(config[str_savedir],'model_best_teacher.state'))
        except KeyboardInterrupt:
            model_student = model_student.cpu().eval()
            model_teacher = model_teacher.cpu().eval()
            torch.save(model_student,              os.path.join(config[str_savedir],'keyboard_interrupt_student.model'),      pickle_module=dill)
            torch.save(model_teacher,              os.path.join(config[str_savedir],'keyboard_interrupt_teacher.model'),      pickle_module=dill)
            torch.save(model_student.state_dict(), os.path.join(config[str_savedir],'keyboard_interrupt_student.state_dict'), pickle_module=dill)
            torch.save(model_teacher.state_dict(), os.path.join(config[str_savedir],'keyboard_interrupt_teacher.state_dict'), pickle_module=dill)
            sak.pickledump(state, os.path.join(config[str_savedir],'keyboard_interrupt.state'), mode='wb')
            raise
        except:
            model_student = model_student.cpu().eval()
            model_teacher = model_teacher.cpu().eval()
            torch.save(model_student,              os.path.join(config[str_savedir],'error_student.model'),      pickle_module=dill)
            torch.save(model_teacher,              os.path.join(config[str_savedir],'error_teacher.model'),      pickle_module=dill)
            torch.save(model_student.state_dict(), os.path.join(config[str_savedir],'error_student.state_dict'), pickle_module=dill)
            torch.save(model_teacher.state_dict(), os.path.join(config[str_savedir],'error_teacher.state_dict'), pickle_module=dill)
            sak.pickledump(state, os.path.join(config[str_savedir],'error.state'), mode='wb')
            raise


# def do_epoch_ssl(model_student: torch.nn.Module, model_teacher: torch.nn.Module, state: dict, config: dict, 
#                  dataloader_labeled:   torch.utils.data.DataLoader, 
#                  dataloader_unlabeled: torch.utils.data.DataLoader, 
#                  criterion: Callable, 
#                  criterion_crossentropy: Callable) -> list:
#     # Log progress
#     max_length = max(len(dataloader_labeled),len(dataloader_unlabeled))
#     batch_loss = np.zeros((max_length,2),dtype='float16')
#     state["moving_dot_product"] = state.get("moving_dot_product",0)

#     # Create transforms
#     if ('data_pre' in config):
#         data_pre     = sak.from_dict(config["data_pre"])
#     if ('augmentation' in config) and model_student.training:
#         augmentation = sak.from_dict(config["augmentation"])
#     if ('data_post' in config):
#         data_post    = sak.from_dict(config["data_post"])

#     # Select iterator decorator
#     train_type = 'Train' if model_student.training else 'Valid'
#     if model_student.training:
#         generator    = get_batch_ssl(dataloader_labeled,dataloader_unlabeled)
#     else:
#         generator    = dataloader_labeled
#     iterator = sak.get_tqdm(generator, config.get('iterator',''), 
#                             desc="({}) Epoch {:>3d}/{:>3d}, Loss {:0.3f}".format(train_type, 
#                                                                                 state['epoch']+1, 
#                                                                                 config['epochs'], np.inf))


#     for i,all_inputs in enumerate(iterator):
#         if model_student.training:
#             inputs_sup,inputs_unsup = all_inputs
#         else:
#             inputs_sup,inputs_unsup = all_inputs,None
#         # Apply data transforms
#         if ('data_pre' in config):
#             data_pre(inputs=inputs_sup)
#             if model_student.training:
#                 data_pre(inputs=inputs_unsup)
#         if ('augmentation' in config) and model_student.training:
#             augmentation(inputs=inputs_sup)
#             if model_student.training:
#                 augmentation(inputs=inputs_unsup)
#         if ('data_post' in config):
#             data_post(inputs=inputs_sup)
#             if model_student.training:
#                 data_post(inputs=inputs_unsup)

#         # UDA
#         if model_student.training:
#             inputs_aug = copy.deepcopy(inputs_unsup)
#             if ('augmentation' in config) and model_student.training:
#                 for _ in range(random.randint(0,config["UDA"].get("max_N",5)-2)):
#                     augmentation(inputs=inputs_aug)
                
#         # Map models to device
#         model_teacher = model_teacher.to(state['device'], non_blocking=True)
#         model_student = model_student.to(state['device'], non_blocking=True)
        
#         # Set gradients to zero
#         if model_teacher.training: state['optimizer_teacher'].zero_grad()
#         if model_student.training: state['optimizer_student'].zero_grad()
        
#         # Predict supervised data
#         inputs_sup            = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_sup.items()}
#         outputs_sup_teacher   = {k: v.cpu() for k,v in model_teacher(inputs_sup).items()}
#         outputs_sup_student   = {k: v.cpu() for k,v in model_student(inputs_sup).items()}
#         inputs_sup            = {k: v.cpu() for k,v in inputs_sup.items()}
        
#         # Predict unsupervised data
#         inputs_unsup          = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_unsup.items()}
#         outputs_unsup_teacher = {k: v.cpu() for k,v in model_teacher(inputs_unsup).items()}
#         inputs_unsup          = {k: v.cpu() for k,v in inputs_unsup.items()}

#         # Predict augmented data
#         inputs_aug            = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_aug.items()}
#         outputs_aug_teacher   = {k: v.cpu() for k,v in model_teacher(inputs_aug).items()}
#         outputs_aug_student   = {k: v.cpu() for k,v in model_student(inputs_aug).items()}
#         inputs_aug            = {k: v.cpu() for k,v in inputs_aug.items()}
        
#         # Supervised loss UDA
#         uda_sup_loss          = criterion(inputs=inputs_sup,outputs=outputs_sup_teacher,state=state)
        
#         # Temperature on unsupervised for UDA
#         outputs_unsup_teacher_temp = {k: v.clone().detach() for k,v in outputs_unsup_teacher.items()}
#         outputs_unsup_teacher_temp["logits"] = outputs_unsup_teacher_temp["logits"]/config["UDA"].get("temperature",0.7)
#         outputs_unsup_teacher_temp["probas"] = torch.sigmoid(outputs_unsup_teacher_temp["logits"])
        
#         # Mask outputs for unsupervised loss
#         largest_probs,_       = torch.max(outputs_unsup_teacher_temp["probas"].detach().flatten(start_dim=1),dim=-1)
#         mask_elements         = torch.ge(largest_probs,config["UDA"].get("mask_threshold",0.6)).float() # GOOD ONE
#         for _ in range(outputs_unsup_teacher_temp["probas"].ndim-1):
#             mask_elements     = mask_elements.unsqueeze(-1)
#         outputs_aug_teacher_mask   = {k: v.clone()*mask_elements for k,v in outputs_aug_teacher.items()}
#         outputs_unsup_teacher_temp = {k: v.clone()*mask_elements for k,v in outputs_unsup_teacher_temp.items()}
#         outputs_unsup_teacher_temp["y"] = outputs_unsup_teacher_temp["probas"]
            
#         # Unsupervised loss UDA
#         uda_unsup_loss        = criterion(inputs=outputs_unsup_teacher_temp,outputs=outputs_aug_teacher_mask,state=state)

#         # Total loss UDA
#         step                  = max_length*state["epoch"] + i
#         uda_weight_decay      = config["UDA"].get("weight_decay",0) # compute weight decay
#         uda_weight            = config["UDA"].get("weight",10)*min(step/config["UDA"].get("steps",1),1)
#         uda_total_loss        = (uda_weight*uda_unsup_loss) + uda_sup_loss + uda_weight_decay
        

#         #########################
#         # Loss on UDA labels, "for backprop" and "For Taylor"
#         outputs_aug_teacher_detached = {k: v.clone().detach() for k,v in outputs_aug_teacher.items()}
#         outputs_aug_teacher_detached["y"] = outputs_aug_teacher_detached["probas"]
#         student_loss          = criterion(inputs=outputs_aug_teacher_detached, outputs=outputs_aug_student,state=state)
#         student_loss_cce_prev = criterion(inputs=inputs_sup, outputs=outputs_sup_student,state=state)

#         # Optimize network's weights
#         # Break early
#         if torch.isnan(student_loss):
#             continue
#             # raise ValueError("Nan loss value encountered. Stopping...")
#         if model_student.training:
#             student_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model_student.parameters(),config["UDA"].get("grad_clip",5.0))
#             state["optimizer_student"].step()
#             if "lr_scheduler_student" in state:
#                 state["lr_scheduler_student"].step()

#         # Apply Exponential Moving Average of student model
#         if config["UDA"].get("ema",0) > 0: 
#             state["ema_model"].update_parameters(model_student)

#         # 2nd call to student: get logits & cross-entropy
#         # Predict supervised data
#         inputs_sup               = {k: v.to(state["device"], non_blocking=True) for k,v in inputs_sup.items()}
#         outputs_sup_student_next = {k: v.cpu() for k,v in model_student(inputs_sup).items()}
#         inputs_sup               = {k: v.cpu() for k,v in inputs_sup.items()}
        
#         # Get supervised loss
#         student_loss_cce_next    = criterion(inputs=inputs_sup, outputs=outputs_sup_student_next,state=state)

#         # Get dot product, moving average of dot product & correct dot product with moving average
#         dot_product              = (student_loss_cce_next - student_loss_cce_prev).detach()
#         moving_dot_product       = state["moving_dot_product"] + 0.01*(dot_product-state["moving_dot_product"])
#         dot_product             -= state["moving_dot_product"]

#         # Final teacher loss
#         crossentropy_loss        = criterion_crossentropy(inputs=outputs_aug_teacher_detached,outputs=outputs_aug_teacher,state=state)
#         teacher_loss             = uda_total_loss + dot_product*crossentropy_loss

#         # Optimize teacher
#         if torch.isnan(teacher_loss):
#             continue
#             # raise ValueError("Nan loss value encountered. Stopping...")
#         if model_teacher.training:
#             teacher_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model_teacher.parameters(),config["UDA"].get("grad_clip",5.0))
#             state["optimizer_teacher"].step()
#             if "lr_scheduler_teacher" in state:
#                 state["lr_scheduler_teacher"].step()
        
#         # Retrieve for printing purposes
#         print_loss = (student_loss.item(),teacher_loss.item())

#         # Accumulate losses
#         batch_loss[i] = print_loss
        
#         # Change iterator description
#         if isinstance(iterator,tqdm.tqdm):
#             if i == max_length-1: batch_print = np.mean(batch_loss,axis=0)
#             else:                 batch_print = print_loss
#             iterator.set_description("({}) Epoch {:>3d}/{:>3d}, Loss {:10.3f}, {:10.3f}".format(train_type, state['epoch']+1, config['epochs'], batch_print[0], batch_print[1]))
#             iterator.refresh()

#     return batch_loss


