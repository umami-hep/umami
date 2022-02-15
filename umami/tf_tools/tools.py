"""Helper tools for tensorflow."""
from tensorflow.keras.callbacks import ReduceLROnPlateau  # pylint: disable=import-error


def GetLRReducer(
    LRR_monitor: str = "loss",
    LRR_factor: float = 0.8,
    LRR_patience: int = 3,
    LRR_verbose: int = 1,
    LRR_mode: str = "auto",
    LRR_cooldown: int = 5,
    LRR_min_lr: float = 0.000001,
    **kwargs,  # pylint: disable=unused-argument
):
    """Call learning rate reducer

    Parameters
    ----------
    LRR_monitor : str, optional
        quantity to be monitored, by default "loss"
    LRR_factor : float, optional
        factor by which the learning rate will be reduced.
        `new_lr = lr * factor`., by default 0.8
    LRR_patience : int, optional
        number of epochs with no improvement after which learning rate
        will be reduced., by default 3
    LRR_verbose : int, optional
        0: quiet, 1: update messages., by default 1
    LRR_mode : str, optional
        one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity, by default "auto"
    LRR_cooldown : int, optional
        number of epochs to wait before resuming normal operation after
        lr has been reduced, by default 5
    LRR_min_lr : float, optional
        Lower bound on the learning rate, by default 0.000001
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    callback
        learning rate reducer callback
    """
    reduce_lr = ReduceLROnPlateau(
        monitor=LRR_monitor,
        factor=LRR_factor,
        patience=LRR_patience,
        verbose=LRR_verbose,
        mode=LRR_mode,
        cooldown=LRR_cooldown,
        min_learning_rate=LRR_min_lr,
    )

    return reduce_lr
