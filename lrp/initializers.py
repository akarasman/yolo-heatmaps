import torch
from .utils import LayerRelevance
            
class YOLOv8RelevanceInitializer(object):

    """
    Assign initial relevance for YOLOv5 model explanation

    Attributes
    ----------

    cls : int

        Index to the class of interest.

    conf_thres : float

        Threshold set for object detection confidence. All output tiles
        with a confidence score lower than this will be truncated to zero

    max_class_only : bool

        Zero all output activations from classes that are not the max.

    contrastive : bool 

        Whether to implement relevance as contrastive or not.

    Methods
    -------

    set_class(cls=None) :
        Set class of interest

    set_prediction(prediction=None) :
        Set prediction

    __call__(prediction : list) :
        Set initial relevance based on prediction made by YOLOv5 model
    
    """

    def __init__(self, cls : int = None, conf : bool = False, 
                 max_class_only : bool = False, contrastive : bool = False):
        
        if contrastive :
            assert cls is not None, "Contrastive implementation of lrp requires target class specification"

        self.cls = cls
        self.conf = conf
        self.max_class_only = max_class_only
        self.contrastive = contrastive

        # prop_to has to do with the YOLOv5 head architecture, more specifically it defines
        # the module numbers that relevance originates from. If this chages the list bellow
        # must be manually changed.
        self.prop_to = [15, 18, 21]
    
    def __call__(self, cls_preds : list):
        
        initial_relevance = []
        norm = 0.0
        for j, cls_pred in enumerate(cls_preds):

            dimensions = cls_pred.size()
            
            # Keep only max class outputs (the rest may be discarded as noise)
            max_class, i = cls_pred.max(dim=1, keepdim=True)
            if self.max_class_only :
                cls_pred = torch.zeros_like(cls_pred).scatter(1, i, max_class)

            # Filter out only class of interest
            if self.cls is not None :
                
                # Construct dual relevance
                if self.contrastive :
                    dual = cls_pred.clone()
                    dual[:, self.cls] = 0.0

                cls_pred[:, :self.cls] = 0.0
                cls_pred[:, self.cls+1:] = 0.0
                
            # Reshape after we're done with processing
            if self.contrastive :
                cls_pred = torch.cat([cls_pred, dual], dim=0)
                
            norm += cls_pred.sum()
            initial_relevance += [ cls_pred ]

        return LayerRelevance(relevance=[ (-1, initial_relevance) ], contrastive=self.contrastive)

