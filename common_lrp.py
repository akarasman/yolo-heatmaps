import torch

def propagate_relevance_SPPF(*args) :

    inverter, mod, relevance = args

    relevance = inverter(mod.cv2, relevance)
    ch = relevance.size(1) // 4

    for i in range(3,0,-1) :
        relevance[:, (i-1)*ch:i*ch, ... ] += inverter(mod.m, relevance[:, i*ch:(i+1)*ch, ... ])
    relevance = inverter(mod.cv1, relevance)
    
    return relevance

def propagate_relevance_Concat(*args) :

    _, mod, relevance = args
    
    relevance_list_out = []
    chunk_e = 0
    for shape in mod.in_shapes :
        chunk_s, chunk_e = chunk_e, chunk_e + shape[mod.d]
        indices = torch.tensor(range(chunk_s, chunk_e)).to(relevance.device)
        relevance_list_out.append(torch.index_select(relevance, mod.d, indices))
    
    return relevance_list_out

def Concat_fwd_hook(m, in_tensors : list, out_tensor : torch.Tensor):

    shapes = [ in_tensor.size(m.d) for in_tensor in in_tensors ]

    setattr(m, "in_shapes", shapes)
    setattr(m, "out_shape", out_tensor.shape)
    
def propagate_relevance_Detect(*args) :

    inverter, mod, relevance = args
    
    relevance_list_out = []
    for m, rel in zip(mod.m, relevance) :
        rel_reshape = torch.cat([ rel[..., i] for i in range(rel.size(-1)) ], dim=1)
        relevance_list_out.append(inverter(m, rel_reshape))
    
    return relevance_list_out

def propagate_relevance_Conv(*args) :

    inverter, mod, relevance = args
    return inverter(mod.conv, relevance)



def propagate_relevance_C3(*args) :

    inverter, mod, relevance = args

    _, relevance = inverter(mod.cv3, relevance)

    c_ = relevance.size(1)

    relevance_mcv1 = relevance[:,:(c_ // 2), ...]
    relevance_cv2 = relevance[:,(c_ // 2):, ...]

    relevance_cv1 = relevance_mcv1
    for m1 in mod.m :
        relevance_cv1 = inverter(m1, relevance_cv1)
    relevance = inverter(mod.cv1, relevance_cv1) + inverter(mod.cv2, relevance_cv2)
    
    return relevance

# Bottleneck only encountered inside of C3
def propagate_relevance_Bottleneck(*args):

    inverter, mod, relevance = args
    return inverter(mod, inverter(mod, relevance))