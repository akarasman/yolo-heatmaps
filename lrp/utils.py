
import torch
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import numpy as np

def flexible_prop(inverse):

    """ Wrapper to help with propagating relevance elegantly """

    def prop_wrapper(layer, relevance, **kwargs) :
        if isinstance(relevance, LayerRelevance) :
            msg = relevance.scatter(-1)
            msg = inverse(layer, msg)
            relevance.gather([(-1, msg)])
        else :
            relevance = inverse(layer, relevance)
        
        return relevance

    return prop_wrapper

def pprint(*args):
    out = [str(argument) + "\n" for argument in args]
    print(*out, "\n")
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))

def load_data():

    df = datasets.load_data_table_15T()

    # Patient-wise train-test-split.
    # Select a number of patients for each class, put all their images in the test set
    # and all other images in the train set. This is the split that is used in the paper to produce the heatmaps.
    test_patients_per_class = 30

    patients_AD = df[df['DX'] == 'Dementia']['PTID'].unique()
    patients_CN = df[df['DX'] == 'CN']['PTID'].unique()

    patients_AD_train, patients_AD_test = train_test_split(patients_AD, test_size=test_patients_per_class,
                                                           random_state=0)
    patients_CN_train, patients_CN_test = train_test_split(patients_CN, test_size=test_patients_per_class,
                                                           random_state=0)

    patients_train = np.concatenate([patients_AD_train, patients_CN_train])
    patients_test = np.concatenate([patients_AD_test, patients_CN_test])

    return datasets.build_datasets(df, patients_train, patients_test, normalize=True)

def scale_mask(mask, shape):

    if shape == mask.shape:
        print("No rescaling necessary.")
        return mask

    nmm_map = np.zeros(shape)
    for lbl_idx in np.unique(mask):
        nmm_map_lbl = mask.copy()
        nmm_map_lbl[lbl_idx != nmm_map_lbl] = 0
        nmm_map_lbl[lbl_idx == nmm_map_lbl] = 1
        zoomed_lbl = zoom(nmm_map_lbl, 1.5, order=3)
        zoomed_lbl[zoomed_lbl != 1] = 0
        remain_diff = np.array(nmm_map.shape) - np.array(zoomed_lbl.shape)
        pad_left = np.array(np.ceil(remain_diff / 2), dtype=int)
        pad_right = np.array(np.floor(remain_diff / 2), dtype=int)
        nmm_map[pad_left[0]:-pad_right[0], pad_left[1]:-pad_right[1], pad_left[2]:-pad_right[2]] += zoomed_lbl * lbl_idx

    return nmm_map

class LayerRelevance(torch.Tensor) :

    """
    LayerRelevance(relevance=None, contrastive=False, print_decimals=5)

    Custom tensor subclass for modeling relevance at any layer

    Attributes
    ----------

    relevance : torch.Tensor
        Input relevance

    contrastive : bool
        Implement relevance propagation as contrastive

    print_decimals : int
        Amount of decimals to use in printing
    
    Methods
    -------

    scatter(which=None, destroy=True)
        Scatters relevance to messages.

    pop_cache(rev_idx)
        Pop cached relevance storage.

    gather(relevance)
        Gather relevance messages.

    snapshot(cached=False)
        Save a copy of relevance as numpy array

    """
    HANDLED_FUNCTIONS = { }

    @staticmethod
    def __new__(cls, relevance=None, contrastive=False, print_decimals=3, *args, **kwargs) :
        
        return super().__new__(cls, [], *args, **kwargs) 
        
    
    def __init__(self, relevance=None, contrastive=False, print_decimals=5):
        
        self.contrastive = contrastive
        self.print_decimals = print_decimals


        # Dual relevance stored as duplicate batch upon initialization
        if isinstance(relevance, list) :
            
            rel_distribution = relevance
        
        elif isinstance(relevance, torch.Tensor) :

            if contrastive :
                rel_distribution = [(-1, torch.cat([relevance[0], relevance[1]], dim=0))]
            else :
                rel_distribution = [(-1, relevance)]

        elif isinstance(relevance, tuple) :

            if contrastive :
                rel_distribution = [(relevance[0], torch.cat([relevance[1], relevance[2]], dim=0))]
            else :
                rel_distribution = [relevance]
        
        elif relevance is None :
            rel_distribution = None

        else :
            raise Exception('Could not convert type {} to input relevance'.
                  format(type(relevance)))

        self.cache = {}
        self.tensor = torch.tensor([])   
        if rel_distribution is not None :
            self.gather(rel_distribution)

    def __str__(self) :

        layer_rel = 0.0
        if self.contrastive :
            if self.tensor.size(0) != 0 :
                layer_rel_primal = self.tensor[0].sum().item()
                layer_rel_dual = self.tensor[1].sum().item()
                cached_rel = { key : [v[0].sum().item(), v[1].sum().item()] \
                               for key, v in self.cache.items() }
                total = layer_rel_primal + layer_rel_dual +\
                        sum([sum(v) for v in cached_rel.values()])
                layer_rel_primal /= total
                layer_rel_dual /= total
                cached_rel =  { key : [v[0]/total, v[1]/total] \
                                for key, v in cached_rel.items() }
                
                layer_rel_primal = round(layer_rel_primal, self.print_decimals)
                layer_rel_dual = round(layer_rel_dual, self.print_decimals)
                layer_rel = 'P:{}/D:{}'.format(layer_rel_primal, layer_rel_dual)
            else :
                cached_rel = { key : [v[0].sum().item(), v[1].sum().item()] \
                               for key, v in self.cache.items() }
                total = sum([ sum(v) for v in cached_rel.values()])
                cached_rel = { key : [ round(v[0] / total, self.print_decimals),
                                       round(v[1] / total, self.print_decimals)] \
                               for key, v in cached_rel.items() }
            cached_rel= ' '.join([ '({}, '.format(key) +
                                   'P:{}'.format(round(v[0], self.print_decimals)) +
                                   '/D:{})'.format(round(v[1], self.print_decimals)) 
                                   for key,v in cached_rel.items() ])

        else :
            layer_rel = self.tensor.sum().item()
            cached_rel = { k : v.sum().item() for k,v in self.cache.items() }
            
            total = layer_rel + sum(cached_rel.values())
            layer_rel = round(layer_rel / total, self.print_decimals)
            cached_rel = { k : round(v / total, self.print_decimals) \
                           for k,v in cached_rel.items()}
            cached_rel = ' '.join(['({}, {})'.format(k,v) for k,v in cached_rel.items()])
        
        return 'LayerRelevance({}, cache={}, contrastive={})'.\
               format(layer_rel, cached_rel, self.contrastive)
        
    
    def scatter(self, which=None, destroy=True):

        """
        Scatters relevance to messages.

        Arguments
        ---------

        which : int
            Layer from which to scatter relevance

        destroy : bool 
            Deletes all relevance from the structure that has been scattered

        Returns
        -------

        torch.Tensor
            Relevance tensor currently scattered

        or when which is None

        list
            List of relevance message tuplse of the form (layer, relevance message)

        """
        
        if which is None :
            relevance = [(-1, self.tensor)] +\
                        [(to, msg) for to, msg in self.cache.items()]

            if destroy :
                del self.tensor, self.cache
                self.tensor = torch.tensor([])
                self.cache = {}
        else :
            if which == -1 :
                relevance = self.tensor
                if destroy :
                    del self.tensor
                    self.tensor = torch.tensor([])
            else :
                relevance = self.cache[which]
                if destroy :
                    del self.cache[which]
        
        return relevance 
    
    def pop_cache(self, rev_idx) :

        """
        Pop cached relevance storage

        Arguments
        ---------

        rev_idx : int 
            Index to the layer of interest to pop cache for, numbered from 
            top layer to back layer.
        
        Returns
        -------

            None
        
        """

        try:
            if self.tensor.nelement() != 0 :
                self.tensor += self.cache[rev_idx]
            else :
                self.tensor = self.cache[rev_idx]
            del self.cache[rev_idx]
        except KeyError :
            pass

    def gather(self, relevance) :

        """
        Gather relevance messages.

        Arguments
        ---------

            relevance : list
                List of relevance messages of the form (layer, relevance message)

        Returns
        -------

            None
        """

        for to, msg in relevance :
                if to != -1 :
                    self.cache[to] = msg
                else :
                    if self.tensor.nelement() == 0 :
                        self.tensor = msg
                    else :
                        self.tensor += msg

    def snapshot(self, cached=False) :

        """
        Save a copy of relevance as numpy array. Using cached=True includes both the
        current layer relevance as well as the cached relevance.

        Arguments
        ---------

            None

        Returns
        -------

            numpy.array
                Relevance at current layer

            dict
                Relevance assigned to each layer
            
        """

        if not cached :
            return self.tensor.cpu().numpy()
        return dict([(-1, self.tensor.cpu().numpy())] +\
                    [(k, v.cpu().numpy()) for k, v in self.cache.items()])

    def __torch_function__(self, func, types, args=(), kwargs=None):
        
        if kwargs is None :
            kwargs = {}
        if func not in LayerRelevance.HANDLED_FUNCTIONS or\
                not all(issubclass(t, (torch.Tensor, LayerRelevance)) for t in types):
            
            args = [a.tensor if hasattr(a, 'tensor') else a for a in args]
            res = func(*args, **kwargs)

            if type(res) is torch.Tensor :
                self.tensor = res
                return self.scatter()
            else :
                return res

        return LayerRelevance.HANDLED_FUNCTIONS[func](*args, **kwargs)
