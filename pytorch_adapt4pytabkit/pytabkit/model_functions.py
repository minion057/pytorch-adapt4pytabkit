from pytabkit.models.training.lightning_modules import TabNNModule
from pytabkit.models.alg_interfaces.base import InterfaceResources

from pytoolbox4dev.base import BaseLogger
logger = BaseLogger(__name__)

def build_empty_tabnn_module(pytabkit_model):
    """
    Build an empty TabNNModule from a given pytabkit model.

    This function extracts the configuration from the provided model
    and initializes a TabNNModule instance without any trained parameters.

    Parameters
    ----------
    pytabkit_model : object
        A pytabkit model instance that provides
        a `get_config()` method returning the model configuration.

    Returns
    -------
    dict
        The model configuration as a dictionary.

    TabNNModule
        An uninitialized (empty) TabNNModule object constructed using
        the retrieved model configuration.

    Notes
    -----
    The returned TabNNModule is created from default configurations only,
    and does not load pretrained weights or training states.
    """
    # from pytabkit.models.sklearn.default_params import DefaultParams
    # https://github.com/dholzmueller/pytabkit/blob/f1b59c2f57fe18ef51b75f890ffd0f77b1d7c6ce/pytabkit/models/sklearn/sklearn_base.py#L96
    model_config = pytabkit_model.get_config() # e.g., == DefaultParams.RealMLP_TD_CLASS
    logger.debug(f'Pytabkit Model config:\n{model_config}')
    return model_config, TabNNModule(**model_config)

def get_fit_params_safe(model_config: dict):
    """
    Safely retrieve the 'fit_params' from a model configuration dictionary.

    Parameters
    ----------
    model_config : dict
        A dictionary that may contain 'fit_params' key with training-related parameters.

    Returns
    -------
    dict or None
        The value of 'fit_params' if it exists in the dictionary, otherwise None.

    Notes
    -----
    This function prevents KeyError by checking if 'fit_params' is available.
    If not available, it prints out an informative message.
    """
    try:
        return model_config['fit_params']
    except KeyError:
        logger.warning('No `fit params` information available.')
        raise KeyError('No `fit params` information available.')

def compare_model_configs(trainer_model, nn_model):
    """
    Compare the configuration dictionaries of a pytabkit model and a PyTorch Lightning model.

    This function iterates through the configuration items of the pytabkit model,
    checking if each key exists and matches in the PyTorch Lightning model's config.
    It prints messages for missing or mismatched configuration entries and
    handles the special case of 'n_epochs' by comparing it with the Lightning module's max_epochs.

    Parameters
    ----------
    trainer_model : object
        The pytabkit model instance with a `get_config()` method returning configuration dictionary.

    nn_model : object
        The PyTorch Lightning model instance containing a `config` attribute (dict)
        and a `progress.max_epochs` attribute for epoch comparison.

    Returns
    -------
    None
        Prints messages indicating missing or mismatched config keys and values.
    """
    def _not_match(k, is_same):
        logger.warning(f'Not matched `{k}` in nn_model.config', debug_mirror_other=True)
        is_same = False
        return is_same
        
    is_same = True
    for k, v in trainer_model.get_config().items():
        if k not in nn_model.config.keys():
            not_found = True
            if k == 'n_epochs':
                not_found = False
                if nn_model.progress.max_epochs != v:  
                    # alternatively compare to nn_model.creator.config['n_epochs']
                    is_same = _not_match(k)
            if not_found: is_same = _not_match(k)
        elif v != nn_model.config[k]: is_same = _not_match(k)
    if not is_same:
        print('The config of the pytabkit model does not match the config of the newly created model.')
        
def compile_and_print_tabnn_model(pytabkit_model, interface_resources_params, 
                                 *compile_args, **compile_kwargs):
    """
    Build, compile, and optionally print a TabNN model based on a pytabkit model.

    This function constructs an empty TabNNModule from the given pytabkit model,
    creates an InterfaceResources instance with the provided parameters, and
    calls the model's compile_model method with the given arguments.
    Optionally prints the compiled model.

    Parameters
    ----------
    pytabkit_model : object
        A pytabkit model instance that supports get_config().

    interface_resources_params : dict
        Dictionary of parameters to initialize InterfaceResources (e.g., n_threads, gpu_devices).

    *compile_args : tuple
        Positional arguments to pass to the compile_model method.

    print_model : bool, optional
        Whether to print the compiled model, by default True.

    **compile_kwargs : dict
        Keyword arguments to pass to the compile_model method.

    Returns
    -------
    dict
        The model configuration as a dictionary.
        
    TabNNModule
        The compiled TabNNModule instance.
    """
    interface_resources = InterfaceResources(**interface_resources_params)
    model_config, nn_model = build_empty_tabnn_module(pytabkit_model)
    try: 
        fit_params = get_fit_params_safe(model_config)
        logger.info(f'Found fit parameters.\n{fit_params}', debug_mirror_other=True)
    except: 
        fit_params = None
    nn_model.compile_model(*compile_args, interface_resources=interface_resources, **compile_kwargs)
    logger.debug(f'Compiled Model:\n{nn_model}')
    compare_model_configs(pytabkit_model, nn_model)
    return model_config, nn_model

def get_leaf_layers(pl_model, print_layers=False):
    """
    Retrieve and optionally print the leaf (lowest-level) layers of a PyTorch Lightning model.

    This function collects all named submodules of the given model, identifies those
    without child modules as leaf layers, and returns both a dictionary of all named
    submodules and a dictionary mapping leaf layer names to their type names.

    Parameters
    ----------
    pl_model : torch.nn.Module
        The PyTorch Lightning model (or any torch.nn.Module) to inspect.

    print_layers : bool, optional
        Whether to print the leaf layers and their types, by default False.

    Returns
    -------
    named_layers : dict
        A dictionary of all named modules within the model {name: module}.

    leaf_types : dict
        A dictionary of leaf layer names to their class type names {name: type_name}.
    """
    leaf_names = []
    named_layers = dict(pl_model.named_modules())
    for name, module in named_layers.items():
        if name == '':  # Exclude the top-level module.
            continue
        if len(list(module.named_children())) == 0:
            # If there are no child modules, consider it as the lowest-level layer.
            leaf_names.append(name)
    leaf_types = {name: type(named_layers[name]).__name__ for name in leaf_names}
    if print_layers:
        for idx, (name, layer_name) in enumerate(leaf_types.items(), 1):
            print(f'{idx:2d}. {name:40s} > {layer_name}')
    return named_layers, leaf_types
