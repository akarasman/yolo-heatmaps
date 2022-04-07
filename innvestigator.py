import torch
from lrp.utils import pprint, LayerRelevance
from lrp.rules import ConvRule, LinearRule
from inverter import Inverter

class InnvestigateModel(torch.nn.Module):

    """
    Generate layerwiser relevance propagation per-pixel explanation for classification
    result of Pytorch model.
    
    (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).

    Attributes
    ----------

    model : torch.nn.Module
        Model to innvestigate

    conv_rule : lrp_rules.ConvRule
        Rule for relevance propagation for convolutional layers

    linear_rule : lrp_rules.LinearRule
        Rule for relevance propagation for convolutional layers

    fwd_hooks : dict 
        Dictionary containing torch.nn.Module to forward hook mappings

    inv_funcs : dict
        Dictionary containing torch.nn.Module to inverse function mappings
    
    contrastive : bool
        Implement relevance propagation as contrastive

    entry_point : torch.nn.Module
        Define entry point in hierarchical network architecture

    pass_not_implemented : bool
        Silent pass layers that have no registered forward hooks 

    device : torch.device
        Device to utilize

    Methods
    -------

    cuda(device : torch.device) :
        Transfer model structure to specified cuda device. If None, the memory
        will be assigned to the first available cuda device.

    cpu(device : torch.device) :
        Use cpu device for computation.

    register_modules(model : torch.nn.Module) :
        Registers model submodules. This will assign a registration number to 
        each submodule in a sequential order.

    register_hooks(model : torch.nn.Module) :
        Registers any necessary forward hooks that save input and output tensors
        for later computation of relevance distribution.

    evaluate(in_tensor : torch.Tensor, **kwargs) -> torch.Tensor :
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron
        per layer.

    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If for example,
        the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, model : torch.nn.Module, 
                 fwd_hooks : dict = {}, inv_funcs : dict = {},
                 contrastive : bool = False, entry_point : torch.nn.Module = None,
                 pass_not_implemented : bool = False, no_children : bool = False,
                 power : int = 1, positive : bool = True, eps : float = 1e-6, 
                 device :  torch.device = torch.device('cpu')):

        super(InnvestigateModel, self).__init__()
        self.model = model
        self.device = device
        self.prediction = None
        self.r_values = None
        self.only_max_score = None
        self.contrastive = contrastive
        self.device = device
        self.conv_rule = ConvRule(power=power,
                                  positive=positive,
                                  eps=eps,
                                  contrastive=contrastive)
        self.linear_rule = LinearRule(power=power,
                                      positive=positive,
                                      eps=eps,
                                      contrastive=contrastive)
        self.linear_rule = LinearRule
        self.layerwise_relevance = []
        self.save_r_values = False

        if entry_point is None :
            entry_point = self.model

        # Initialize the 'Relevance Propagator' with the chosen rule.
        # This will be used to back-propagate the relevance values
        # through the layers in the innvestigate method.
        self.inverter = Inverter(linear_rule = self.linear_rule,
                                 conv_rule = self.conv_rule,
                                 pass_not_implemented=pass_not_implemented,
                                 device=self.device)

        self.register_new_modules(fwd_hooks, inv_funcs)
        
        # Parsing the individual model layers
        
        self.register_hooks(entry_point)
        self.register_modules(entry_point, no_children)

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
        return super(InnvestigateModel, self).cuda(device)

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
        return super(InnvestigateModel, self).cpu()

    def register_modules(self, entry_point : torch.nn.Module, no_children : bool = False) :

        if no_children :
            setattr(entry_point, 'reg_num', len(self.inverter.module_list))
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
                mod.register_forward_hook(self.inverter.get_layer_fwd_hook(mod))
                continue
            
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
        The innvestigate wrapper returns the same prediction as the
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
        self.prediction = self.model(in_tensor, **kwargs)

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

    def innvestigate(self, in_tensor=None, initializer=None):

        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        
        Arguments
        ---------

        in_tensor : torch.Tensor
            Input for which to evaluate the LRP algorithm. If input is None, the
            last evaluation is used. If no evaluation has been performed since
            initialization, an error is raised.
        
        rel_for_class : int
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

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)            

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            
            if initializer is not None :
                relevance = initializer(self.prediction)
            else :
                relevance = LayerRelevance(self.prediction)
            
            # List to save relevance distributions per layer
            self.r_values = [relevance]
            #print('Start', 'Relevance', relevance)
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance.pop_cache(layer.reg_num)
                self.r_values.append(relevance)
                relevance = self.inverter(layer, relevance)
                #print(layer.reg_num, type(layer), 'Relevance', relevance)

            if self.save_r_values :
                self.r_values.append(relevance.snapshot())
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return relevance

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

        return self.model.forward(in_tensor)

    def extra_repr(self):

        """
        Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        return self.model.extra_repr()
