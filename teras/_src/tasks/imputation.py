import keras
from teras._src.api_export import teras_export


@teras_export("teras.tasks.Imputer")
class Imputer:
    """
    Imputer task class used to impute missing data using the trained `model`
    instance.

    Args:
        model: keras.Model, trained instance of a keras model that will be
            used to impute missing data.
            Currently, `teras` offers `GAIN` and `PCGAIN` architectures for
            imputation.
        data_transformer: Instance of a relevant data transformer that is
            used during the data transformation step before training the
            respective architecture.
    """
    def __init__(self,
                 model: keras.Model,
                 data_transformer=None):
        self.model = model
        self.data_transformer = data_transformer

    def impute(self, x, reverse_transform=True, batch_size=None,
               verbose="auto", steps=None, callbacks=None):
        """
        Imputes the missing data.
        It exposes all the arguments taken by the `predict` method.

        Args:
            x: pd.DataFrame, dataset with missing values.
            reverse_transform: bool, default True, whether to reverse
                transformed the raw imputed data to original format.

        Returns:
            Imputed data in the original format.
        """
        if reverse_transform:
            if self.data_transformer is None:
                raise ValueError(
                    "To reverse transform the raw imputed data, "
                    "`data_transformer` must not be None. "
                    "Please pass the instance of `DataTransformer` class used "
                    "to transform the input data as argument to this `impute` "
                    "method. \n"
                    "Or alternatively, you can set `reverse_transform` "
                    "parameter to False, and manually reverse transform the "
                    "generated raw data to original format using the "
                    "`reverse_transform` method of `DataTransformer` instance.")
        x_imputed = self.model.predict(x,
                                       batch_size=batch_size,
                                       verbose=verbose,
                                       steps=steps,
                                       callbacks=callbacks)
        if reverse_transform:
            x_imputed = self.data_transformer.reverse_transform(x_imputed)
        return x_imputed
