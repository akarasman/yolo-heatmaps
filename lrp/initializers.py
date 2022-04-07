import torch
from lrp.utils import LayerRelevance

class StandardInitializer(object):

    """
    Assign initial relevance for YOLOv5 model explanation

    Attributes
    ----------

    rel_for_class : int
        
        Index to the class of interest.
    
    """

    def __init__(self, rel_for_class=None):
        self.rel_for_class = rel_for_class

    def __call__(self, prediction):

        if self.rel_for_class is None:

            # Default behaviour is innvestigating the output
            # on an arg-max-basis, if no class is specified.
            org_shape = prediction.size()
            prediction = prediction.view(org_shape[0], -1)
            max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
            only_max_score = torch.zeros_like(self.prediction).to(self.device)
            only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
            relevance_tensor = only_max_score.view(org_shape)

        else:

            org_shape = prediction.size()
            prediction = prediction.view(org_shape[0], -1)
            only_max_score = torch.zeros_like(self.prediction).to(self.device)
            only_max_score[:, self.rel_for_class] += self.prediction[:, self.rel_for_class]
            relevance_tensor = only_max_score.view(org_shape)
        
        return LayerRelevance(relevance_tensor)
            
class YOLOv5Initializer(object):

    """
    Assign initial relevance for YOLOv5 model explanation

    Attributes
    ----------

    rel_for_class : int

        Index to the class of interest.

    box : tuple / list

        Bounding box for which to generate explanation.

    conf_thres : float

        Threshold set for object detection confidence. All output tiles
        with a confidence score lower than this will be truncated to zero

    max_class_only : bool

        Zero all output activations from classes that are not the max.

    contrastive : bool 

        Whether to implement relevance as contrastive or not.

    Methods
    -------

    set_box(box=None) : 
        Sets bounding box for explanation to specified list or tuple

    set_class(rel_for_class=None) :
        Set class of interest

    set_prediction(prediction=None) :
        Set prediction

    __call__(prediction : list) :
        Set initial relevance based on prediction made by YOLOv5 model
    
    """

    def __init__(self, rel_for_class : int = None, box : list = None,
                 conf : bool = False, max_class_only : bool = False,
                 contrastive : bool = False):
        
        if contrastive :
            assert (rel_for_class is not None, 
                    "Contrastive implementation of lrp requires target class specification")

        self.rel_for_class = rel_for_class
        self.conf = conf
        self.max_class_only = max_class_only
        self.contrastive = contrastive
        self.box=box

        # prop_to has to do with the YOLOv5 head architecture, more specifically it defines
        # the module numbers that relevance originates from. If this chages the list bellow
        # must be manually changed.
        self.prop_to_ = [17, 20, 23]

    def set_box(self, box=None) :

        """ Set new box for explanation """
        self.box = box
        return self

    def set_class(self, rel_for_class=None) :

        """ Set new class of interest """
        self.rel_for_class = rel_for_class
        return self
    
    def set_prediction(self, prediction=None) :

        """ Set prediction """
        self.rel_for_class = prediction
        return self
    
    def __call__(self, prediction):

        nc = prediction[0].shape[-1] - 5  # number of classes
        initial_relevance = []
        
        norm = 0
        for to, x in zip(self.prop_to_, prediction):  # image index, image inference

            xs = x.size()
            x = x.clone()

            # Problematic case of zero batches
            if not xs[0] :
                continue

            # Flattened shape leads to more efficient initialization
            x = x.view(-1, nc+5)            

            # Limit outputs to box
            if self.box is not None :
                i = ((x[:, 0] < self.box[0]) + (x[:, 1] < self.box[1]) +
                     (x[:, 0] > self.box[2]) + (x[:, 1] > self.box[3]))

                x[i, 4:] = torch.zeros_like(x[i, 4:])

            # Multiply by object existance prior probabillity
            if self.conf :
                x[:, 5:] = x[:, [4]] * x[:, 5:]
            
            # Zero x,y,h,w, confidence
            x[:, :5] = torch.zeros_like(x[:, :5])
            
            # Keep only max class outputs (the rest may be discarded as noise)
            max_class, i = x[:, 5:].max(dim=1, keepdim=True)
            if self.max_class_only :
                x[:, 5:] = torch.zeros_like(x[:, 5:]).scatter(-1, i, max_class)

            # Filter out only class of interest
            if self.rel_for_class is not None :
                
                # Construct dual relevance
                if self.contrastive :
                    dual = x.clone()
                    dual[:, self.rel_for_class] = torch.zeros(dual.size(0))
                    max_class_dual, i_dual = dual[:, 5:].max(dim=1, keepdim=True)
                    dual[:,5:] = torch.zeros_like(dual[:, 5:]).scatter(-1, i_dual, max_class_dual)
                

                x[:, 5:5+self.rel_for_class] = torch.zeros_like(x[:, 5:5+self.rel_for_class])
                x[:, self.rel_for_class+6:] = torch.zeros_like(x[:, self.rel_for_class+6:])
            
            # Reshape after we're done with processing
            if self.contrastive :
                x = torch.cat([x.view(xs), dual.view(xs)], dim=0)
            else :
                x = x.view(xs)
            
            norm += x.sum()

            initial_relevance.append((to, x))

        return LayerRelevance(relevance=initial_relevance, contrastive=self.contrastive)

class SSDInitializer(object):

    def __init__(self, rel_for_class : int = None, box : list = None,
                 conf_thres : float = 0.25, max_class_only : bool = False,
                 contrastive : bool = False):
        
        if contrastive :
            assert (rel_for_class is not None, 
                    "Contrastive implementation of lrp requires target class specification")

        self.rel_for_class = rel_for_class
        self.conf_thres = conf_thres
        self.max_class_only = max_class_only
        self.contrastive = contrastive
        self.box=box

    def set_class(self, rel_for_class=None) :

        """ Set new class of interest """
        self.rel_for_class = rel_for_class
        return self
    
    def set_prediction(self, prediction=None) :

        """ Set prediction """
        self.rel_for_class = prediction
        return self
    
    def __call__(self, prediction):

        _, conf = prediction

        confs = conf.size()
        conf = conf.clone()

        # Problematic case of zero batches
        if not confs[0] :
            return          
        
        # Keep only max class outputs (the rest may be discarded as noise)
        max_class, i = conf[..., 1:].max(dim=1, keepdim=True)
        if self.max_class_only :
            conf[..., 1:] = torch.zeros_like(conf[..., 1:]).scatter(-1, i, max_class)
        
        # Apply confidence threshold, truncating outputs bellow this to zero
        if self.conf_thres > 0:
            cc = (conf[..., [0]] * max_class < self.conf_thres)[...,0]
            x[cc] = torch.zeros_like(conf[cc])
        
        # Zero x,y,h,w, confidence
        conf[..., 0] = torch.zeros_like(conf[..., 0])

        # Filter out only class of interest
        if self.rel_for_class is not None :
            
            # Construct dual relevance
            if self.contrastive :
                dual = torch.cat( (x[:, :5+self.rel_for_class],
                                    torch.zeros_like(x[:, [self.rel_for_class]]),
                                    x[:, 6+self.rel_for_class:]), dim=-1)
            
            conf[..., :self.rel_for_class] = torch.zeros_like(conf[..., :self.rel_for_class])
            conf[..., self.rel_for_class+1:] = torch.zeros_like(conf[..., self.rel_for_class+1:])

    
        # Reshape after we're done with processing
        if self.contrastive :
            x = torch.cat([x.view(xs), dual.view(xs)], dim=0)
        else :
            x = x.view(xs)

        norm += x.sum() # Update normalization denominator
        initial_relevance.append((to, x))

        # Normalize s.t sum of relevance is equal to 1
        initial_relevance = [ (k, x) for k, x in initial_relevance]

        return LayerRelevance(relevance=initial_relevance, contrastive=self.contrastive)
