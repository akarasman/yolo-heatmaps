from numpy import iterable
import torch
import torch.nn.functional as F
from .utils import pprint, flexible_prop


def winner_takes_all(relevance_in : torch.Tensor, in_shape : iterable, indices : torch.Tensor ) -> torch.Tensor :

    """
    Implements winner takes-all scheme for re-distibution of relevance
    for a max pooling layer

    Arguments
    ---------

    relevance_in : torch.Tensor
        Incoming relevance from upper layers.

    in_shape : list or tuple
        Shape of module input.

    indices : torch.Tensor
        Indexes of selected (max) features.
    
    Returns
    -------

    relevance_out : torch.Tensor
        Relevance redistributed to lower layer.

    """ 
    # (REAL SLOW, MAKE THIS FASTER !)
    
    _, _, H, W = relevance_in.size()
    N = H * W
    relevance_out = []

    for rin in relevance_in :
        rout = torch.zeros(in_shape).flatten()
        relevance_flat = rin.flatten()
        
        for i, idx in enumerate(indices.flatten()):
            rout[idx + (i // N) * N] += relevance_flat[i]

        relevance_out.append(rout.view(in_shape))
    
    return torch.cat(relevance_out, dim=0)

def conv_nd_fwd_hook(m, in_tensor, out_tensor):

    """ Default n-dimensional convolution forward hook """
    
    setattr(m, "in_tensor", in_tensor[0])
    setattr(m, "out_tensor", out_tensor)

def max_pool_nd_fwd_hook(m, in_tensor, out_tensor):

    """ Default n-dimensional max pool forward hook """
    
    cache = m.return_indices
    _, indices = F.max_pool2d(in_tensor[0], kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                              dilation=m.dilation, return_indices=True, ceil_mode=m.ceil_mode)
    setattr(m, "indices", indices)
    setattr(m, 'out_shape', out_tensor.size())
    setattr(m, 'in_shape', in_tensor[0].size())

def upsample_fwd_hook(m, in_tensor, out_tensor):

    """ Default up-sampling forward hook """
    
    setattr(m, 'in_dim', len(in_tensor[0].shape))
    setattr(m, 'out_shape', out_tensor.shape)

def linear_fwd_hook(m, in_tensor, out_tensor):

    """ Default Linear layer forward hook  """

    setattr(m, "in_tensor", in_tensor[0])
    setattr(m, "out_shape", list(out_tensor.size()))

def silent_pass(m, in_tensor, out_tensor):

    """ Silent forward hook that saves nothing """

    pass

def LogSoftmax_inverse(relevance : torch.Tensor, warn : bool = True) -> torch.Tensor :

    """
    Inversion of LogSoftmax layer

    Arguments
    ---------

    relevance : torch.Tensor 
        Input relavance

    warn : bool
        Display warning message when applied

    Returns
    -------

    torch.Tensor
        Output relevance
    """

    if relevance.sum() < 0:
        relevance[relevance == 0] = -1e6
        relevance = relevance.exp()
        if warn :
            pprint("WARNING: LogSoftmax layer was "
                   "turned into probabilities.")
    
    return relevance

@flexible_prop
def max_pool_nd_inverse(layer, relevance_in : torch.Tensor, indices : torch.Tensor = None, 
                        max : bool = False) -> torch.Tensor :

    """
    Inversion of LogSoftmax layer

    Arguments
    ---------

    relevance : torch.Tensor 
        Input relavance

    indices : torch.Tensor
        Maximum feature indexes obtained when max pooling

    max : bool
        Implement winner takes all scheme in relevance re-distribution

    Returns
    -------

    torch.Tensor
        Output relevance
    """
    
    if indices is None :
        indices = layer.indices
    
    out_shape = layer.out_shape
    bs = relevance_in.size(0)
    relevance_in = torch.cat([r.view(out_shape) for r in relevance_in ], dim=0)

    indices = torch.cat([indices] * bs, dim=0)
    
    return ( winner_takes_all(relevance_in, layer.in_shape, layer.indices) 
             if max else relevance_in )


@flexible_prop
def upsample_inverse(layer, relevance : torch.Tensor) -> torch.Tensor :

    """
    Inversion of upsample layer

    Arguments
    ---------

    relevance : torch.Tensor 
        Input relavance

    Returns
    -------

    torch.Tensor
        Output relevance

    ATTENTION : Currently only 'nearest' upsampling method is invertable
    """

    invert_upsample = {
        1 : F.avg_pool1d,
        2 : F.avg_pool2d,
        3 : F.avg_pool3d
    } [layer.in_dim - 2]

    if layer.mode != 'nearest' :
        raise NotImplementedError("Upsample layer must be in 'nearest' mode ")
    relevance_in = torch.cat([r.view(layer.out_shape) for r in relevance], dim=0)

    if isinstance(layer.scale_factor, float):
        ks = int(layer.scale_factor)
    elif isinstance(layer.scale_factor, tuple):
        ks = tuple([ int(s) for s in layer.scale_factor ])

    inverted = invert_upsample(relevance_in, kernel_size=ks, stride=ks)
    inverted *= ks**2 # Normalizing constant

    return inverted
