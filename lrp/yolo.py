import torch
from .utils import pprint
from .initializers import YOLOv8RelevanceInitializer
from .rules import ConvRule, LinearRule
from .inverter import Inverter
from .common import ( prop_Concat, Concat_fwd_hook, prop_Detect, 
                      prop_C3, prop_Conv, prop_Bottleneck, prop_DFL,
                      prop_SPPF, SPPF_fwd_hook, prop_C2f )

from ultralytics.nn.modules import block, conv, head

class YOLOv8LRP(torch.nn.Module):

    """
    Generate layerwiser relevance propagation per-pixel explanation for classification
    result of Pytorch model.
    
    (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).

    Attributes
    ----------

    model : torch.nn.Module
        Model to explain

    conv_rule : lrp_rules.ConvRule
        Rule for relevance propagation for convolutional layers

    linear_rule : lrp_rules.LinearRule
        Rule for relevance propagation for convolutional layers
    
    contrastive : bool
        Implement relevance propagation as contrastive

    device : torch.device
        Device to utilize

    Methods
    -------

    cuda(device : torch.device) :
        Transfer model structure to specified cuda device. If None, the memory
        will be assigned to the first available cuda device.

    cpu(device : torch.device) :
        Use cpu device for computation.

    evaluate(in_tensor : torch.Tensor, **kwargs) -> torch.Tensor :
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron
        per layer.

    ATTENTION:
        Currently, generating heatmaps for a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If for example,
        the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, model : torch.nn.Module, 
                 contrastive : bool = False,  power : int = 1, positive : bool = True, eps : float = 1e-6, 
                 device :  torch.device = torch.device('cpu')):

        super(YOLOv8LRP, self).__init__()
        self.model = model
        self.device = device
        self.prediction = None
        self.r_values = None
        self.only_max_score = None
        self.contrastive = contrastive
        self.device = device
        conv_rule = ConvRule(power=power,
                             positive=positive,
                             eps=eps,
                             contrastive=contrastive)
        linear_rule = LinearRule(power=power,
                                 positive=positive,
                                 eps=eps,
                                 contrastive=contrastive)
        self.linear_rule = LinearRule
        self.layerwise_relevance = []
        self.save_r_values = False


        # Initialize the 'Relevance Propagator' with the chosen rule.
        # This will be used to back-propagate the relevance values
        # through the layers.
        self.inverter = Inverter(linear_rule=linear_rule,
                                 conv_rule=conv_rule,
                                 pass_not_implemented=True,
                                 device=self.device)
        self.register_new_modules({ conv.Concat : Concat_fwd_hook,
                                    block.SPPF : SPPF_fwd_hook }, 
                                  { block.C3 : prop_C3,
                                    block.Conv : prop_Conv,
                                    head.Detect : prop_Detect,
                                    block.Bottleneck : prop_Bottleneck,
                                    conv.Concat : prop_Concat,
                                    block.SPPF : prop_SPPF,
                                    block.DFL : prop_DFL,
                                    block.C2f : prop_C2f })
        
        # Parsing the individual model layers
        self.register_hooks(self.model.model.model)
        self.register_modules(self.model.model.model)

        self.relevance_cache = {}

    def cuda(self, device : torch.device = None):

        """
        Transfer model structure to specified cuda device. If None, the memory
        will be assigned to the first available cuda device.

        Arguments
        ---------

        device : torch.device
            Device to put memory into.

        Returns
        -------

            None
        """

        self.device = torch.device("cuda", device)
        self.inverter.device = self.device
        return super(YOLOv8LRP, self).cuda(device)

    def cpu(self):

        """
        Use cpu device for computation.

        Arguments
        ---------

            None

        Returns
        -------

            None
        """
        self.device = torch.device("cpu", 0)
        self.inverter.device = self.device
        return super(YOLOv8LRP, self).cpu()

    def register_modules(self, entry_point : torch.nn.Module, no_children : bool = False) :

        if no_children :
            self.inverter.module_list.append(entry_point)
        else :
            for mod in entry_point.children():
                setattr(mod, 'reg_num', len(self.inverter.module_list))
                
                self.inverter.module_list.append(mod)

    def register_hooks(self, parent_module : torch.nn.Module):

        """
        Registers any necessary forward hooks that save input and output tensors
        for later computation of relevance distribution.
        
        Args
        ----

        parent_module : torch.nn.Module
            Model to register hooks for.

        Returns
        -------

            None

        """

        for mod in parent_module.children():
            
            
            if list(mod.children()):
                self.register_hooks(mod)
            
            if len(mod._forward_hooks) == 0:
                mod.register_forward_hook(self.inverter.get_layer_fwd_hook(mod))

            # Special case for ReLU layer
            if isinstance(mod, torch.nn.ReLU) or isinstance(mod, torch.nn.modules.activation.ReLU):
                mod.register_backward_hook(self.relu_hook_function)
    
    def register_new_modules(self, fwd_hooks : dict = {}, inv_funcs : dict = {}) :

        for mod, fwd_hook in fwd_hooks.items() :
            self.inverter.register_fwd_hook(mod, fwd_hook)
        
        for mod, inv_func in inv_funcs.items() :
            self.inverter.register_inv_func(mod, inv_func)
    
    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):

        """
        If there is a negative gradient, change it to zero.
        """

        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):

        """
        The explanation wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.

        Arguments
        ---------

        in_tensor : torch.Tensor
            Model input to pass through the pytorch model.

        Returns
        -------

        torch.Tensor
            Model output
        """

        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor : torch.Tensor, **kwargs) -> torch.Tensor :

        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.

        Arguments
        ---------
        
        in_tensor : torch.Tensor
            New input for which to predict an output.

        Returns
        -------
        
        torch.Tensor
            Model prediction
        """

        # Reset module list. In case the structure changes dynamically,
        # the module list is tracked for every forward passs.
        # self.inverter.reset_module_list()
        
        self.prediction = self.model.model(in_tensor.unsqueeze(0), **kwargs)
        return self.prediction

    def get_r_values_per_layer(self):

        """
        Get relevance snapshots per layer in the network.

        Arguments
        --------

        None

        Returns
        -------

        list
            list of relevance snapshots
        
        """

        if self.r_values is None:
            pprint("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
        return self.r_values

    def explain(self, frame, cls= None, conf=False, max_class_only=True, contrastive=False,
                b1=0.5, b2=0.5):

        """
        Method for generating an explanatort heatmap for the model with the LRP rule chosen at
        the initialization of the module.
        
        Arguments
        ---------

        frame : torch.Tensor
            Input frame for which to evaluate the LRP algorithm. 
        
        cls : int
            Index of the class for which the relevance distribution is to be analyzed.
            If None, the 'winning' class is used for indexing.

        Returns
        -------

        LayerRelevance
            Model output and relevances of nodes in the input layer
        """
        
        if self.r_values is not None:
            for elt in self.r_values:
                del elt
            self.r_values = None
        
        if isinstance(cls, str):
            cls = list(self.model.names.values()).index(cls)

        with torch.no_grad():
    
            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            
            initializer = YOLOv8RelevanceInitializer(cls=cls, max_class_only=max_class_only,
                                                     conf=conf, contrastive=contrastive)
        
            self.model.model.predict(frame.unsqueeze(0))

            cls_preds = [ conv[-1].out_tensor.sigmoid() for conv in self.model.model.model[-1].cv3 ]
            relevance = initializer(cls_preds=cls_preds)
            
            # List to save relevance distributions per layer
            self.r_values = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance.pop_cache(layer.reg_num)
                self.r_values.append(relevance)
                relevance = self.inverter(layer, relevance)

            if self.save_r_values :
                self.r_values.append(relevance.snapshot())
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            lrp_out = relevance.scatter(-1)

            if contrastive :
                lrp_p = lrp_out[[0]]
                lrp_d = lrp_out[[1]]
                top_5_percent_threshold = torch.quantile(lrp_p, 0.98)
                outlier_mask = lrp_p < top_5_percent_threshold
                lrp_p = torch.where(outlier_mask, lrp_p, torch.tensor(0.0))
                top_5_percent_threshold = torch.quantile(lrp_p, 0.98)
                outlier_mask = lrp_d < top_5_percent_threshold
                lrp_d = torch.where(outlier_mask, lrp_d, torch.tensor(0.0))
                
                explanation  = (b1*lrp_p - b2*lrp_d).sum(dim=1)[0]
            else :
                lrp_p = lrp_out[[0]]
                lrp_p = lrp_p.sum(dim=1)[0]
                top_5_percent_threshold = torch.quantile(lrp_p, 0.98)
                outlier_mask = lrp_p < top_5_percent_threshold
                explanation = torch.where(outlier_mask, lrp_p, torch.tensor(0.0))
                
                
            return explanation

    def forward(self, in_tensor : torch.Tensor) -> torch.Tensor :

        """
        Evaluates model on a given input tensor.

        Arguments
        ---------

        in_tensor : torch.Tensor
            Model input tensor.

        Returns
        -------

        torch.Tensor
            Model output tensor.
        """

        return self.model(in_tensor)

    def extra_repr(self):

        """
        Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        return self.model.extra_repr()
