def assert_fitted(func):
    def wrapper(*args, **kwargs):
        self, *_ = args
        if not self._fitted:
            raise AssertionError(
                f"`{self.__dict__.__class__}` has not been fitted yet. "
                f"Please fit by calling the `fit` method before accessing the "
                f"{func.__name__} method.")
        return func(*args, **kwargs)
    return wrapper
