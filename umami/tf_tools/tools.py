"""Helper tools for tensorflow."""
from tensorflow.keras.callbacks import ReduceLROnPlateau  # pylint: disable=import-error

from umami.train_tools.configuration import NNStructureConfig


def get_learning_rate_reducer(nn_structure: NNStructureConfig):
    """Call learning rate reducer.

    Parameters
    ----------
    nn_structure: umami.train_tools.configuration.NNStructureConfig
        Loaded train_config nn_structure part

    Returns
    -------
    callback
        learning rate reducer callback
    """
    reduce_lr = ReduceLROnPlateau(
        monitor=nn_structure.lrr_monitor,
        factor=nn_structure.lrr_factor,
        patience=nn_structure.lrr_patience,
        verbose=nn_structure.lrr_verbose,
        mode=nn_structure.lrr_mode,
        cooldown=nn_structure.lrr_cooldown,
        min_learning_rate=nn_structure.lrr_min_learning_rate,
    )

    return reduce_lr
