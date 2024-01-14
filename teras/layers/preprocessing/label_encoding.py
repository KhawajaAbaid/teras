import keras
from keras import ops
from teras.utils.types import Number


@keras.saving.register_keras_serializable(package="teras.layers.preprocessing")
class LabelEncoding(keras.layers.Layer):
    """
    Standalone Encoding layer
    Essentially it acts as a Label encoder.
    It maps the numeric values (int/float) to their indices.

    Args:
        inputs: ``KerasTensor``,
            Inputs with categorical features that should be encoded

        concatenate_numerical_features: ``bool``, default False,
            Whether to concatenate numerical features
            to the encoded categorical features.
    """
    def __init__(self,
                 features_metadata: dict,
                 concatenate_numerical_features: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.features_metadata = features_metadata
        self.features_metadata["categorical"] = self.features_metadata["categorical"]
        self.concatenate_numerical_features = concatenate_numerical_features

        self._categorical_features_idx = list(map(lambda x: x[0], self.features_metadata["categorical"].values()))
        self._num_categorical_features = len(self._categorical_features_idx)
        self._numerical_features_idx = list(self.features_metadata["numerical"].values())
        self._num_numerical_features = len(self._numerical_features_idx)
        self._num_features = self._num_numerical_features + self._num_categorical_features

        # a boolean list where each index corresponds to the index of a feature in the dataset
        # and the boolean value at that index indicates whether the feature is categorical
        self._is_categorical = [feat_id in self._categorical_features_idx
                                for feat_id in range(self._num_features)]

        self.lookup_tables = self._get_lookup_tables()

    def _get_lookup_tables(self):
        """Lookup tables for each categorical feature"""
        lookup_tables = []
        for feature_name, (feature_idx, vocabulary) in self.features_metadata["categorical"].items():
            # Lookup Table to convert int/float values to integer indices
            if isinstance(vocabulary[0], Number):
                lookup = keras.layers.IntegerLookup(vocabulary=vocabulary,
                                                    mask_token=None,
                                                    output_mode="int",
                                                    )
            else:
                raise TypeError("`LabelEncoding` layer can only encode values of type int "
                                f"but received type: {type(vocabulary[0])} for feature {feature_name}.")
            lookup_tables.append(lookup)
        return lookup_tables

    def call(self, inputs):
        encoded_features = ops.array([])
        categorical_counter = 0
        for feature_idx in range(self._num_features):
            feature = ops.expand_dims(inputs[:, feature_idx], axis=1)
            if self._is_categorical[feature_idx]:
                lookup = self.lookup_tables[categorical_counter]
                feature = ops.cast(lookup(feature), dtype="float32")
                encoded_features = ops.append(encoded_features, feature)
                categorical_counter += 1
            # the feature is numerical and if the flag is set, we concatenate it,
            # otherwise just ignore it
            elif self.concatenate_numerical_features:
                encoded_features = ops.append(encoded_features, feature)
        encoded_features = ops.reshape(ops.squeeze(encoded_features),
                                       (-1,
                                        self._num_features if self.concatenate_numerical_features
                                        else self._num_categorical_features)
                                       )

        return encoded_features

    def get_config(self):
        config = super().get_config()
        new_config = {'features_metadata': self.features_metadata,
                      'concatenate_numerical_features': self.concatenate_numerical_features
                      }
        config.update(new_config)
        return config
