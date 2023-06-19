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
            attrs_temp = {name: value for name, value in attrs_temp.items() if not name.startswith("__")}
            attrs.update(attrs_temp)
            if current_class.__base__.__name__ == "BaseConfig":
                current_class = None
            else:
                current_class = current_class.__base__
        return attrs


class UserModifiableBaseConfig:
    """
    Base Config class for all the Config classes whose instance attributes
    are expected to be modified by the users.
    It contains all the common functionality.
    """
    def to_dict(self):
        attrs = self.__dict__
        attrs = {name: value for name, value in attrs.items() if not name.startswith("__")}
        return attrs


class FitConfig(UserModifiableBaseConfig):
    """
    Config class for model's fit method.
    It is to be used when a function wraps a model and calls model's fit method
    such that the model's `fit` method isn't exposed to user directly and
    hence user cannot specify the parameters.
    """
    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.epochs = 1
        self.verbose = 'auto'
        self.callbacks = None
        self.validation_split = 0.0
        self.validation_data = None
        self.shuffle = True
        self.class_weight = None
        self.sample_weight = None
        self.initial_epoch = 0
        self.steps_per_epoch = None
        self.validation_steps = None
        self.validation_batch_size = None
        self.validation_freq = 1
        self.max_queue_size = 10
        self.workers = 1
        self.use_multiprocessing = False
