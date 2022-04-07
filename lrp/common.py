import torch

from torch.nn.functional import max_unpool2d, max_pool2d
from lrp.utils import LayerRelevance

def prop_SPPF(*args):

    inverter, mod, relevance = args

    #relevance = torch.cat([r.view(mod.m.out_shape) for r in relevance ], dim=0)
    bs = relevance.size(0)
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

    # Because concatenate
    slices = relevance.scatter(-1).split(mod.in_shapes, dim=mod.d)
    relevance.gather([(to, msg) for to, msg in zip(mod.f, slices)])

    return relevance

def prop_Detect(*args):

    inverter, mod, relevance = args
    relevance_out = []

    scattered = relevance.scatter()
    for m, rel in zip(mod.m, scattered[1:]):
        to, msg = rel
        msg = torch.cat([msg[..., i] for i in range(msg.size(-1))], dim=1)
        to = to if to != mod.reg_num else -1
        out = inverter(m, msg)
        relevance_out.append((to, out))

    relevance.gather(relevance_out)

    return relevance


def prop_Conv(*args):

    inverter, mod, relevance = args
    return inverter(mod.conv, relevance)


def prop_C3(*args):

    inverter, mod, relevance = args
    msg = relevance.scatter(which=-1)

    c_ = msg.size(1)

    msg_cv1 = msg[:, : (c_ // 2), ...]
    msg_cv2 = msg[:, (c_ // 2) :, ...]

    for m1 in mod.m:
        msg_cv1 = inverter(m1, msg_cv1)
    
    msg = inverter(mod.cv1, msg_cv1) + inverter(mod.cv2, msg_cv2)

    relevance.gather([(-1, msg)])

    return relevance

def prop_Bottleneck(*args):

    inverter, mod, relevance_in = args

    ar = mod.cv2.conv.out_tensor.abs()
    ax = mod.cv1.conv.in_tensor.abs()

    relevance = relevance_in #* ar / (ax + ar)
    relevance = inverter(mod.cv1, relevance)
    relevance = inverter(mod.cv2, relevance)
    relevance = relevance #+ relevance_in * ax / (ax + ar)

    return relevance
