class CTGAN:
    raise ImportError(
        "JAX backend doesn't provide a Keras MODEL implementation for `CTGAN`. "
        "Rather it offers `CTGANTrainer` based on functional approach which "
        "can be accessed through `teras.trainers` package. \n"
        "Alternatively, if you really want to use a model based implementation "
        "for `CTGAN`, you can use the `tensorflow` or `torch` backend."
    )
