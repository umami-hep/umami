from tensorflow.keras.callbacks import ReduceLROnPlateau


def GetLRReducer(
    LRR_monitor: str = "loss",
    LRR_factor: float = 0.8,
    LRR_patience: int = 3,
    LRR_verbose: int = 1,
    LRR_mode: str = "auto",
    LRR_cooldown: int = 5,
    LRR_min_lr: float = 0.000001,
    **kwargs,
):
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
