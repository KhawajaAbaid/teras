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

