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
