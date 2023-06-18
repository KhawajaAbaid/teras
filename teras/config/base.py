class BaseConfig:
    """
    Base Config class for all the Config classes.
    It contains all the common functionality.
    """
    @classmethod
    def to_dict(cls):
        attrs = {}
        current_class = cls
        while current_class:
            attrs_temp = vars(cls)
            attrs_temp = {name: value for name, value in attrs.items() if not name.startswith("__")}
            attrs.update(attrs_temp)
            if current_class.__base__.__name__ == "BaseConfig":
                current_class = None
            else:
                current_class = current_class.__base__
        return attrs


class FitConfig(BaseConfig):
    """
    Config class for model's fit method.
    It is to be used when a function wraps a model and calls model's fit method
    such that the model's `fit` method isn't exposed to user directly and
    hence user cannot specify the parameters.
    """
    batch_size = None
    epochs = 1
    verbose = 'auto'
    callbacks = None
    validation_split = 0.0
    validation_data = None
    shuffle = True
    class_weight = None
    sample_weight = None
    initial_epoch = 0
    steps_per_epoch = None
    validation_steps = None
    validation_batch_size = None
    validation_freq = 1
    max_queue_size = 10
    workers = 1
    use_multiprocessing = False
