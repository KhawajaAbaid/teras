# Contains code for teras v0.2.0


try:
    import tensorflow
except ModuleNotFoundError:
    raise ("teras.legacy requires tensorflow. Please install it before "
           "using it.")


try:
    import tensorflow_probability
except ModuleNotFoundError:
    raise ("teras.legacy requires tensorflow_probability. "
           "Please install it before using it.")
