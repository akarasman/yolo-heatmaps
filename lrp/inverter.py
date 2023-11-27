# trunk-ignore(black-py)
import torch
from torch.nn import Conv1d, Conv2d, Conv3d, MaxPool1d, MaxPool2d, MaxPool3d, Linear, Upsample
from .utils import Flatten
from .inverter_util import ( upsample_inverse, max_pool_nd_inverse,
                            max_pool_nd_fwd_hook, conv_nd_fwd_hook, linear_fwd_hook,
                            upsample_fwd_hook, silent_pass )

FWD_HOOK = { torch.nn.MaxPool1d : max_pool_nd_fwd_hook,
             torch.nn.MaxPool2d : max_pool_nd_fwd_hook,
             torch.nn.MaxPool3d : max_pool_nd_fwd_hook,
             torch.nn.Conv1d : conv_nd_fwd_hook,
             torch.nn.Conv2d : conv_nd_fwd_hook,
             torch.nn.Conv3d : conv_nd_fwd_hook,
             torch.nn.Linear : linear_fwd_hook,
             torch.nn.Upsample : upsample_fwd_hook,
             torch.nn.BatchNorm1d : silent_pass,
             torch.nn.BatchNorm2d : conv_nd_fwd_hook,
             torch.nn.BatchNorm3d : silent_pass,
             torch.nn.ReLU : silent_pass,
             torch.nn.modules.activation.ReLU : silent_pass,
             torch.nn.ELU : silent_pass,
             Flatten : silent_pass,
             torch.nn.Dropout : silent_pass,
             torch.nn.Dropout2d : silent_pass,
             torch.nn.Dropout3d : silent_pass,
             torch.nn.Softmax : silent_pass,
             torch.nn.LogSoftmax : silent_pass,
             torch.nn.Sigmoid : silent_pass,
             torch.nn.SiLU :  silent_pass }

# Rule-independant inversion functions
IDENTITY_MAPPINGS = ( torch.nn.BatchNorm1d,
                      torch.nn.BatchNorm2d,
                      torch.nn.BatchNorm3d,
                      torch.nn.ReLU,
                      torch.nn.modules.activation.ReLU,
                      torch.nn.ELU,
                      Flatten,
                      torch.nn.Dropout,
                      torch.nn.Dropout2d,
                      torch.nn.Dropout3d,
                      torch.nn.Softmax,
                      torch.nn.LogSoftmax,
                      torch.nn.Sigmoid, 
                      torch.nn.SiLU )

def module_tracker(fwd_hook_func):
 
    """
    Wrapper for tracking the layers throughout the forward pass.

    Arguments
    ---------
        
        fwd_hook_func : function
            Forward hook function to be wrapped.

    Returns
    -------

        function :
            Wrapped hook function

    """

    def hook_wrapper(layer, *args):
        return fwd_hook_func(layer, *args)

    return hook_wrapper

class Inverter(torch.nn.Module):
    
    """
    Class for computing the relevance propagation and supplying the necessary forward hooks for all layers.

    Attributes
    ----------

    linear_rule : LinearRule
        Propagation rule to use for linear layers

    conv_rule : ConvRule
        Propagation rule for convolutional layers

    pass_not_implemented : bool
        Silent pass layers that have no registered forward hooks

    device : torch.device
        Device to put relevance data

    Methods
    -------

        Propagates incoming relevance for the specified layer, applying any
        necessary inversion functions along the way.
    
    """

    # Implemented rules for relevance propagation.
    def __init__(self, linear_rule=None, conv_rule=None, pass_not_implemented=False,
                 device=torch.device('cpu'),):

        self.device = device
        self.warned_log_softmax = False
        self.linear_rule = linear_rule
        self.conv_rule = conv_rule
        self.fwd_hooks = FWD_HOOK
        self.inv_funcs= {}
        self.pass_not_implemented = pass_not_implemented
        self.module_list = []
    
    def register_fwd_hook(self, module, fwd_hook):

        """
        Register forward hook function to module.
        """

        if module in self.fwd_hooks.keys():
            print('Warning: Replacing previous fwd hook registered for {}'.
                  format(module))
        
        self.fwd_hooks[module] = fwd_hook
    
    def register_inv_func(self, module, inv_func):

        """
        Register inverse function to module.
        """

        if module in self.inv_funcs.keys():
            print('Warning: Replacing previous inverse registered for {}'.
                  format(module))

        self.inv_funcs[module] =  inv_func

    def get_layer_fwd_hook(self, layer) :

        """
        Interface for getting any layer's forward hook
        """

        try :
            return self.fwd_hooks[type(layer)]
        except :
            if self.pass_not_implemented :
                return silent_pass
            
            raise \
            NotImplementedError('Forward hook for layer type \"{}\" not implemented'.
                                format(type(layer)))

    def invert(self, layer : torch.nn.Module, relevance : torch.Tensor, **kwargs) -> torch.Tensor :

        """
        This method computes the backward pass for the incoming relevance
        for the specified layer.

        Arguments
        ---------

            layer : torch.nn.Module
                Layer to propagate relevance through. Can be Conv1d, Conv2d or
                any combination thereof in a higher level module.
            
            relevance : torch.Tensor
                Incoming relevance from higher up in the network.

        Returns
        ------

            torch.Tensor :
                Redistributed relevance going to the lower layers in the network.
                
        """
        
        if isinstance(layer, (Conv1d, Conv2d, Conv3d)):
            if self.conv_rule is None :
                raise Exception('Model contains conv layers but the conv rule was not set !')
            return self.conv_rule(layer, relevance, **kwargs)
        elif isinstance(layer, (MaxPool1d, MaxPool2d, MaxPool3d)):
            return max_pool_nd_inverse(layer, relevance)
        elif isinstance(layer, Linear) :
            if self.linear_rule is None :
                raise Exception('Model contains linear layers but the linear rule was not set !')
            return self.linear_rule(layer, relevance.tensor, **kwargs)
        elif isinstance(layer, Upsample):
            return upsample_inverse(layer, relevance)
        elif isinstance(layer, torch.nn.modules.container.Sequential):
            for l in layer[::-1] :
                relevance = self.invert(l, relevance)
            return relevance
        elif type(layer) in IDENTITY_MAPPINGS :
            return relevance
        elif hasattr(layer, 'propagate'):
            return layer.propagate(self, relevance)
        else :
            try :
                return self.inv_funcs[type(layer)](self, layer, relevance, **kwargs) 
            except KeyError :
                raise NotImplementedError(f'Relevance propagation not implemented for layer type {type(layer)}')

    def __call__(self, layer : torch.nn.Module, relevance : torch.Tensor, **kwargs) -> torch.Tensor :

        """ Wrapper for invert method """
        return self.invert(layer, relevance, **kwargs)