# trunk-ignore(isort)
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, Conv3d
from .utils import LayerRelevance

class LinearRule(object):

    """
    WARNING: This utility has not been tested and may contain significant errors

    LinearRule(power : int = 1, eps : float = 1e-06, positive : bool = True, 
               contrastive : bool =True,)

    Apply layerwise relevance propagation rule to linear layer. Implemented using
    the procedure described here : 
    
    http://iphome.hhi.de/samek/pdf/MonXAI19.pdf

    Attributes
    ----------

    contrastive : bool
        Compute dual relevance for contrastive

    power : int / float
        Exponent to apply to input / weights

    eps : float
        Small number added to denominator to avoid divide by zero 

    positive : bool
        Truncate negative activations of prev layer to zero


    Methods
    -------
    
    propagate(module : torch.nn.Module, relevance : torch.Tensor) -> torch.Tensor
        Propagates incoming relevance through module and redistributes
        it to lower layers.

    """ 

    def __init__(self, power : int = 1, eps : float = 1e-06, positive : bool = True, 
                 contrastive : bool =True,):
        
        self.power = power
        self.eps = eps
        self.positive = positive
        self.contrastive = contrastive

    def __call__(self, module, relevance,):

        """ Wrapper for propagate, makes rule object callable """

        return self.propagate(module, relevance)
    
    def propagate(self, module : torch.nn.Module, relevance : torch.Tensor,) -> torch.Tensor :

        """

        Propagate incoming relevance through module and redistribute it to lower layers.

        Arguments
        ---------

        module : torch.nn.Module (Linear)
            Module through which relevance is propagated.

        relevance : torch.Tensor
            LayerRelevance tensor or simple tensor containing upper layer relevance.

        Returns
        -------
        
        torch.Tensor 
            Re-distributed relevance. If input is tensor output will also be tensor, and if
            it is LayerRelevance output will also be LayerRelevance tensor.

        ATTENTION : Each time LayerRelevance is "propagated" through a module it essentialy
        "morphs" into the output relevance. This is to say that any previous relevance info
        is not saved and is discarded. If for the sake of visualizing the relevance distribu-
        tion as it backpropagates through the network you need to keep a cache of each layer
        relevance you can do so by using the cache() method.

        """

        # Implementation-wise this method is just a wrapper for the private
        # propagation utillity that ensures functionallity for both LayerRelevance.
        # objects and tensors.
        if isinstance(relevance, LayerRelevance):
            msg = relevance.scatter(-1)
            msg = self.__propagate(module, msg)
            relevance.gather([(-1, msg)])
        else:
            relevance = self.__propagate(module, relevance)

        return relevance

    def __get_fwd_step(self,):

        """ Dimension non-specific forward function  """

        def linear_wrapper(in_tensor, w, **kwargs):
            if self.contrastive:
                x = torch.cat([in_tensor] * 2, dim=0)
            else:
                x = in_tensor
            return F.linear(x, w, **kwargs)

        return linear_wrapper

    def __get_bwd_step(self,):

        """ Dimension non-specific inverse function  """

        def linear_wrapper(relevance_in, w, **kwargs):
            return F.linear(relevance_in, w.t, **kwargs)

        return linear_wrapper

    def __propagate(self, module, relevance_in,):

        linear_fwd = self.__get_fwd_step()
        linear_bwd = self.__get_bwd_step()

        # Pre-process : Apply power to weights/input
        x = module.in_tensor
        x = x.pow(self.power)
        w = module.weight.pow(self.power)

        # Compute forward activation with modified weights (step 1)
        z = linear_fwd(x, w, bias=None)
        z = z + torch.sign(z) * self.eps
        relevance_in[z == 0] = 0
        z[z == 0] = 1

        # Divide incoming relevance by activation (step 2)
        s = relevance_in / z

        # Compute gradient of s in terms of its input, which the equivalent of applying
        # the de-convolution operation (see PyTorch doc) (step 3)  
        c = linear_bwd(s, w, bias=None)

        # Multiply by input (step 4)
        relevance_out = c * x

        return relevance_out
class ConvRule(object):

    """

    ConvRule(power : int = 1, positive : bool = True, eps : float = 1e-6, 
             contrastive : bool = True,) 

    Apply layerwise relevance propagation rule to convolutional layers.
    Implemented efficiently using procedure described here : 
    
    http://iphome.hhi.de/samek/pdf/MonXAI19.pdf

    Attributes
    ----------

    power : int / float
        Exponent to apply to input / weights
    
    positive : bool
        Truncate negative activations of prev layer to zero

    eps : float
        Small number added to denominator to avoid divide by zero 

    contrastive : bool
        Compute dual relevance for contrastive

    Methods
    -------
    
    propagate(module : torch.nn.Module, relevance : torch.Tensor) -> torch.Tensor
        Propagates incoming relevance through module and redistributes
        it to lower layers.

    """ 

    def __init__(self, power : int = 1, eps : float = 1e-6, positive : bool = True,
                 contrastive : bool = True,) :
        
        self.power = power
        self.eps = eps
        self.positive = positive
        self.contrastive = contrastive

    def __call__(self, module, relevance,) ->  torch.Tensor :
        
        """ Wrapper for propagate, makes rule object callable """

        return self.propagate(module, relevance)

    def propagate(self, module : torch.nn.Module, relevance : torch.Tensor,) -> torch.Tensor :

        """

        Propagate incoming relevance through module and redistribute it to lower layers.

        Arguments
        ---------

        module : torch.nn.Module (Conv1d, Conv2d or Conv3d)
            Conv module through which relevance is propagated.

        relevance : torch.Tensor
            LayerRelevance tensor or simple tensor containing upper layer relevance.

        Returns
        -------
        
        torch.Tensor 
            Re-distributed relevance. If input is tensor output will also be tensor, and if
            it is LayerRelevance output will also be LayerRelevance tensor.

        ATTENTION : Each time LayerRelevance is "propagated" through a module it essentialy
        "morphs" into the output relevance. This is to say that any previous relevance info
        is not saved and is discarded. If for the sake of visualizing the relevance distribu-
        tion as it backpropagates through the network you need to keep a cache of each layer
        relevance you can do so by using the cache() method.

        """

        # Implementation-wise this method is just a wrapper for the private propagation 
        # utillity that ensures functionallity for both LayerRelevance.objects and tensors.

        if isinstance(relevance, LayerRelevance):
            msg = relevance.scatter(-1)
            msg = self.__propagate(module, msg)
            relevance.gather([(-1, msg)])
        else:
            relevance = self.__propagate(module, relevance)

        return relevance

    def __get_fwd_step(self, m):

        """ Dimension non-specific forward function  """

        try:
            conv = {Conv1d: F.conv1d, Conv2d: F.conv2d, Conv3d: F.conv3d}[type(m)]
        except:
            raise Exception("Layer must be one of {}".format((Conv1d, Conv2d, Conv3d)))
        
        # The only function of the following wrapper is to duplicate the input
        # tensor. This is necessary for computing contrastive relevance propagation
        # efficiently.
        def conv_wrapper(in_tensor, **kwargs):
            if self.contrastive:
                x = torch.cat([in_tensor, in_tensor], dim=0)
            else:
                x = in_tensor
            return conv(x, **kwargs)

        return conv_wrapper

    def __get_bwd_step(self, m):

        """ Dimension non-specific inverse function  """

        try:
            inv_conv = {
                Conv1d: F.conv_transpose1d,
                Conv2d: F.conv_transpose2d,
                Conv3d: F.conv_transpose3d,
            }[type(m)]
        except:
            raise Exception("Layer must be one of {}".format((Conv1d, Conv2d, Conv3d)))

        def inv_conv_wrapper(relevance_in, **kwargs):
            return inv_conv(relevance_in, **kwargs)

        return inv_conv_wrapper

    def __propagate(self, module, relevance_in):

        """ Implementation of 4-step relevance propagation procedure """

        relevance_in = torch.cat([ r.view_as(module.out_tensor) for r in relevance_in ], dim=0)
        conv_fwd = self.__get_fwd_step(module)
        conv_bwd = self.__get_bwd_step(module)

        with torch.no_grad():
            
            # Pre-process : Apply power to weights/input, discard negative activations
            x = torch.cat( [module.in_tensor.clone()] * relevance_in.size(0), dim=0)
            w = module.weight.clone()
            if self.positive :
                x = x.clamp(min=0)
                w = w.clamp(min=0)
            x = x.pow(self.power)
            w = w.pow(self.power)

            # Compute forward activation with modified weights (step 1)
            
            z = conv_fwd(x, weight=w, bias=None, stride=module.stride,
                         padding=module.padding, groups=module.groups,) #
            z = z + torch.sign(z) * self.eps
            relevance_in[z == 0] = 0
            z[z == 0] = 1

            # Divide incoming relevance by activation (step 2)
            s = relevance_in / z
            
            # Compute gradient of s in terms of its input, which the equivalent of applying
            # the de-convolution operation (see PyTorch doc) (step 3)  
            if module.stride != (1, 1) :
                
                # When stride is not equal to 1 there may be some discrepancy between
                # the input and output sizes, the following code fixes this by manually
                # calculating the new H, w dimensions and adding valid padding at backward
                # step       
                _, _, H, W = relevance_in.size()
                Hnew = (H - 1) * module.stride[0] - 2*module.padding[0] +\
                        module.dilation[0]*(module.kernel_size[0]-1) +\
                        module.output_padding[0]+1
                Wnew = (W - 1) * module.stride[1] - 2*module.padding[1] +\
                        module.dilation[1]*(module.kernel_size[1]-1) +\
                        module.output_padding[1]+1
                _, _, Hin, Win = x.size()

                cp = conv_bwd(s, weight=w, bias=None, padding=module.padding,
                            output_padding=(Hin-Hnew, Win-Wnew), stride=module.stride,
                            dilation=module.dilation, groups=module.groups,)
            else:
                cp = conv_bwd(s, weight=w, bias=None, padding=module.padding,
                            stride=module.stride, groups=module.groups,)

            
            # Multiply by input (step 4)
            relevance_out = cp*x

            return relevance_out
