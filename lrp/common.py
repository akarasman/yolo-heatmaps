import torch

from torch.nn.functional import max_unpool2d, max_pool2d
from .utils import LayerRelevance, get_dummy_summation_conv_layer
from .inverter_util import conv_nd_fwd_hook


def prop_SPPF(*args):

    inverter, mod, relevance = args
    
    relevance = inverter(mod.cv2, relevance)
    msg = relevance.scatter(which=-1)
    ch = msg.size(1) // 4
    
    r3 = msg[:, 3*ch:4*ch, ...] 
    r2 = msg[:, 2*ch:3*ch, ...] + r3   
    r1 = msg[:, ch:2*ch, ...] + r2
    rx = msg[:, :ch, ...] + r1
    
    msg = inverter(mod.cv1, rx)
    relevance.gather([(-1, msg)])

    return relevance

def SPPF_fwd_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):

    x = m.cv1(in_tensor[0])
    y1, idx1 = max_pool2d(x, kernel_size = m.m.kernel_size,
                             stride=m.m.stride,
                             padding=m.m.padding,
                             dilation=m.m.dilation,
                             return_indices=True,
                             ceil_mode=m.m.ceil_mode)
    y2, idx2 = max_pool2d(y1, kernel_size = m.m.kernel_size,
                              stride=m.m.stride,
                              padding=m.m.padding,
                              dilation=m.m.dilation,
                              return_indices=True,
                              ceil_mode=m.m.ceil_mode)
    y3, idx3 = max_pool2d(y2, kernel_size = m.m.kernel_size,
                              stride=m.m.stride,
                              padding=m.m.padding,
                              dilation=m.m.dilation,
                              return_indices=True,
                              ceil_mode=m.m.ceil_mode)
    setattr(m, "indices", [idx1, idx2, idx3])

def Concat_fwd_hook(m, in_tensors: torch.Tensor, out_tensor: torch.Tensor):

    shapes = [in_tensor.shape[m.d] for in_tensor in in_tensors[0]]

    setattr(m, "in_shapes", shapes)
    setattr(m, "out_shape", out_tensor.shape)

def prop_Concat(*args):

    _, mod, relevance = args

    slices = relevance.scatter(-1).split(mod.in_shapes, dim=mod.d)
    relevance.gather([(to, msg) for to, msg in zip(mod.f, slices)])

    return relevance

def prop_Detect(*args):

    inverter, mod, relevance = args
    relevance_out = []
    
    _, scattered = relevance.scatter()[0]
    prop_to = [21, 18, 15][::-1]
    for i, rel in enumerate(scattered):
        relevance_out.append((prop_to[i], inverter(mod.cv3[i], rel)))
        inverter(mod.cv3[i], rel)

    relevance.gather(relevance_out)
    return relevance


def prop_Conv(*args):

    inverter, mod, relevance = args
    return inverter(mod.conv, relevance)


def prop_C3(*args):

    inverter, mod, relevance = args
    msg = relevance.scatter(which=-1)

    msg = inverter(mod.cv3, msg) 

    c_ = msg.size(1)

    msg_cv1 = msg[:, : (c_ // 2), ...]
    msg_cv2 = msg[:, (c_ // 2) :, ...]

    for m1 in mod.m:
        msg_cv1 = inverter(m1, msg_cv1)
    
    msg = inverter(mod.cv1, msg_cv1) + inverter(mod.cv2, msg_cv2)

    relevance.gather([(-1, msg)])

    return relevance

def prop_Bottleneck(*args):

    inverter, mod, relevance = args

    msg = relevance
    c = msg.shape[1]
    
    dummy_conv = get_dummy_summation_conv_layer(c)
    x = mod.cv1.conv.in_tensor
    

    y = mod.cv2.conv.out_tensor + mod.cv1.conv.in_tensor
    xy = torch.concatenate((x,y), dim=1)
    conv_nd_fwd_hook(dummy_conv, [xy], (x+y))
    
    msg = inverter(dummy_conv, msg)
    msg_in = msg[:, :c, ...]
    msg_res = msg[:, c:, ...]
    msg_res = inverter(mod.cv1, msg_res)
    msg_res = inverter(mod.cv2, msg_res)
    msg = msg_in + msg_res
    
    return msg


def prop_C2f(*args):
    # Extract relevant tensors from the module
    
    inverter, mod, relevance = args
    msg = relevance.scatter(which=-1)
    msg_cv2 = inverter(mod.cv2, msg)
    msg_m = list(msg_cv2.chunk(msg_cv2.size(1) // mod.c, 1))

    # Relevance propagation through the bottleneck blocks (m)
    for i, m_block in enumerate(mod.m[::-1]):
        msg_m[-(i+2)] += inverter(m_block, msg_m[-(i+1)])
    msg_cv1 = torch.cat(msg_m[:2], axis=1)
    msg = inverter(mod.cv1, msg_cv1)
    relevance.gather([(-1, msg)])
    return relevance


def prop_DFL(*args) :
    
    _, _, a = mod.in_shape
    inverter, mod, relevance = args
    relevance = inverter(relevance.unsqueeze(0), mod.conv.weight.data).transpose(2, 1)
    relevance = torch.cat([ relevance[0,:,:,ai].flatten().unsqueeze(-1) for ai in range(a) ], axis=-1).unsqueeze(0)
    
    return relevance