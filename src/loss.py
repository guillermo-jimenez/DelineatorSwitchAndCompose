from typing import Tuple, Iterable
import numpy as np
import torch 
import torch.nn
import sak
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import Conv3d
from torch.nn import Sigmoid
from torch.nn import L1Loss
from torch.nn import MSELoss
from torch.nn import BCELoss

from sak.__ops import required
from sak.__ops import check_required

class F1InstanceLossNoGateNoSoft(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', weight: Iterable = None, threshold: float = 10, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x
        
        # Define auxiliary loss
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.sigmoid = Sigmoid()
        
        # Define convolutional operation
        self.prewitt = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.prewitt.parameters():
            param.requires_grad = False

        # Override weights
        self.prewitt.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.prewitt.weight[c,c, 0] = -1.
            self.prewitt.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt = self.prewitt.to(target.device)

        # Retrieve boundaries
        input_boundary = self.prewitt(input).abs()
        target_boundary = self.prewitt(target).abs()

        # Sum of elements alongside the spatial dimensions
        input_elements = torch.flatten(input_boundary, start_dim=2).sum(-1)/4
        target_elements = torch.flatten(target_boundary, start_dim=2).sum(-1)/4

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            input_elements = input_elements*self.weight
            target_elements = target_elements*self.weight

        # Instance loss
        loss = (target_elements-input_elements).abs()/(target_elements+input_elements)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)

class F1InstanceLossNewGateNoSoft(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', weight: Iterable = None, threshold: float = 10, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x
        
        # Define auxiliary loss
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.sigmoid = Sigmoid()
        
        # Define convolutional operation
        self.prewitt = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.prewitt.parameters():
            param.requires_grad = False

        # Override weights
        self.prewitt.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.prewitt.weight[c,c, 0] = -1.
            self.prewitt.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt = self.prewitt.to(target.device)

        # Retrieve boundaries
        input_boundary = self.prewitt(input).abs()
        target_boundary = self.prewitt(target).abs()

        # Sum of elements alongside the spatial dimensions
        input_elements = torch.flatten(input_boundary, start_dim=2).sum(-1)/4
        target_elements = torch.flatten(target_boundary, start_dim=2).sum(-1)/4

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            input_elements = input_elements*self.weight
            target_elements = target_elements*self.weight

        # Hack to get whether target_elements or input_elements is larger
        gate = self.sigmoid((target_elements-input_elements)*self.threshold)

        # Basic metrics
        truepositive  = (target_elements-gate*(target_elements-input_elements)).abs()
        falsepositive = (1-gate)*(input_elements-target_elements).abs()
        falsenegative = gate*(target_elements-input_elements).abs()

        # "F1 loss"
        loss = 1-(2*truepositive + 1)/(2*truepositive + falsepositive + falsenegative + 1)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)

class F1InstanceLossNoGateSoft(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', weight: Iterable = None, threshold: float = 10, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x
        
        # Define auxiliary loss
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.sigmoid = Sigmoid()
        
        # Define convolutional operation
        self.prewitt = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.prewitt.parameters():
            param.requires_grad = False

        # Override weights
        self.prewitt.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.prewitt.weight[c,c, 0] = -1.
            self.prewitt.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt = self.prewitt.to(target.device)

        # Soften input
        input = self.sigmoid((input-0.5)*self.threshold)
        target = self.sigmoid((target-0.5)*self.threshold)

        # Retrieve boundaries
        input_boundary = self.prewitt(input).abs()
        target_boundary = self.prewitt(target).abs()

        # Obtain sigmoid-ed input and target
        input_boundary  = self.sigmoid((input_boundary-0.5)*self.threshold) # Rule of thumb for dividing the classes as much as possible
        target_boundary = self.sigmoid((target_boundary-0.5)*self.threshold) # Rule of thumb for dividing the classes as much as possible

        # Sum of elements alongside the spatial dimensions
        input_elements = torch.flatten(input_boundary, start_dim=2).sum(-1)/4
        target_elements = torch.flatten(target_boundary, start_dim=2).sum(-1)/4

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            input_elements = input_elements*self.weight
            target_elements = target_elements*self.weight

        # Instance loss
        loss = (target_elements-input_elements).abs()/(target_elements+input_elements)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)

class F1InstanceLossNewGateSoft(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', weight: Iterable = None, threshold: float = 10, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x
        
        # Define auxiliary loss
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.sigmoid = Sigmoid()
        
        # Define convolutional operation
        self.prewitt = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.prewitt.parameters():
            param.requires_grad = False

        # Override weights
        self.prewitt.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.prewitt.weight[c,c, 0] = -1.
            self.prewitt.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt = self.prewitt.to(target.device)

        # Soften input
        input = self.sigmoid((input-0.5)*self.threshold)
        target = self.sigmoid((target-0.5)*self.threshold)

        # Retrieve boundaries
        input_boundary = self.prewitt(input).abs()
        target_boundary = self.prewitt(target).abs()

        # Obtain sigmoid-ed input and target
        input_boundary  = self.sigmoid((input_boundary-0.5)*self.threshold) # Rule of thumb for dividing the classes as much as possible
        target_boundary = self.sigmoid((target_boundary-0.5)*self.threshold) # Rule of thumb for dividing the classes as much as possible

        # Sum of elements alongside the spatial dimensions
        input_elements = torch.flatten(input_boundary, start_dim=2).sum(-1)/4
        target_elements = torch.flatten(target_boundary, start_dim=2).sum(-1)/4

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            input_elements = input_elements*self.weight
            target_elements = target_elements*self.weight

        # Hack to get whether target_elements or input_elements is larger
        gate = self.sigmoid((target_elements-input_elements)*self.threshold)

        # Basic metrics
        truepositive  = (target_elements-gate*(target_elements-input_elements)).abs()
        falsepositive = (1-gate)*(input_elements-target_elements).abs()
        falsenegative = gate*(target_elements-input_elements).abs()

        # "F1 loss"
        loss = 1-(2*truepositive + 1)/(2*truepositive + falsepositive + falsenegative + 1)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)

class F1InstanceLossOldGateNoSoft(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', weight: Iterable = None, threshold: float = 10, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x
        
        # Define auxiliary loss
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.sigmoid = Sigmoid()
        
        # Define convolutional operation
        self.prewitt = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.prewitt.parameters():
            param.requires_grad = False

        # Override weights
        self.prewitt.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.prewitt.weight[c,c, 0] = -1.
            self.prewitt.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt = self.prewitt.to(target.device)

        # Retrieve boundaries
        input_boundary = self.prewitt(input).abs()
        target_boundary = self.prewitt(target).abs()

        # Sum of elements alongside the spatial dimensions
        input_elements = torch.flatten(input_boundary, start_dim=2).sum(-1)/4
        target_elements = torch.flatten(target_boundary, start_dim=2).sum(-1)/4

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            input_elements = input_elements*self.weight
            target_elements = target_elements*self.weight

        # Hack to get whether target_elements or input_elements is larger
        gate = self.sigmoid(target_elements-input_elements)

        # Basic metrics
        truepositive  = (target_elements-gate*(target_elements-input_elements)).abs()
        falsepositive = (1-gate)*(input_elements-target_elements).abs()
        falsenegative = gate*(target_elements-input_elements).abs()

        # "F1 loss"
        loss = 1-(2*truepositive + 1)/(2*truepositive + falsepositive + falsenegative + 1)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)


class F1InstanceLossOldGateSoft(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', weight: Iterable = None, threshold: float = 10, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x
        
        # Define auxiliary loss
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.sigmoid = Sigmoid()
        
        # Define convolutional operation
        self.prewitt = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.prewitt.parameters():
            param.requires_grad = False

        # Override weights
        self.prewitt.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.prewitt.weight[c,c, 0] = -1.
            self.prewitt.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt = self.prewitt.to(target.device)

        # Soften input
        input = self.sigmoid((input-0.5)*self.threshold)
        target = self.sigmoid((target-0.5)*self.threshold)

        # Retrieve boundaries
        input_boundary = self.prewitt(input).abs()
        target_boundary = self.prewitt(target).abs()

        # Obtain sigmoid-ed input and target
        input_boundary  = self.sigmoid((input_boundary-0.5)*self.threshold) # Rule of thumb for dividing the classes as much as possible
        target_boundary = self.sigmoid((target_boundary-0.5)*self.threshold) # Rule of thumb for dividing the classes as much as possible

        # Sum of elements alongside the spatial dimensions
        input_elements = torch.flatten(input_boundary, start_dim=2).sum(-1)/4
        target_elements = torch.flatten(target_boundary, start_dim=2).sum(-1)/4

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            input_elements = input_elements*self.weight
            target_elements = target_elements*self.weight

        # Hack to get whether target_elements or input_elements is larger
        gate = self.sigmoid(target_elements-input_elements)

        # Basic metrics
        truepositive  = (target_elements-gate*(target_elements-input_elements)).abs()
        falsepositive = (1-gate)*(input_elements-target_elements).abs()
        falsenegative = gate*(target_elements-input_elements).abs()

        # "F1 loss"
        loss = 1-(2*truepositive + 1)/(2*truepositive + falsepositive + falsenegative + 1)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)


