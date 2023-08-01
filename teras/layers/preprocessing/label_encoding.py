import tensorflow as tf
from tensorflow import keras
from teras.utils.types import Number


@keras.saving.register_keras_serializable(package="teras.layers.preprocessing")
class LabelEncoding(keras.layers.Layer):
    """
    Standalone Encoding layer
    Essentially it acts as a Label encoder.
    It maps the numeric values (int/float) to their indices.

    Args:
        inputs: ``numpy ndarray`` or ``tf.Tensor``,
            Inputs with categorical features that should be encoded

        concatenate_numerical_features: ``bool``, default False,
            Whether to concatenate numerical features
            to the encoded categorical features.

        keep_features_order: ``bool``, default True,
            If True, the returning dataset will have the features in the same
            order as the input.
            If False, the returning dataset, will have k categorical features
            first and then the numerical features follow in the order they appear
            in the dataset i.e. first come, first served essentially.

            This parameter will only be taken into account when the
            `concatenate_numerical_features` is set to True,
            as otherwise returning dataset will only contain categorical features
            in order that they appear in the inputs.
    """
    def __init__(self,
                 features_metadata: dict,
                 concatenate_numerical_features: bool = False,
                 keep_features_order: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.features_metadata = features_metadata
        self.features_metadata["categorical"] = self.features_metadata["categorical"]
        self.concatenate_numerical_features = concatenate_numerical_features
        self.keep_features_order = keep_features_order

        self._categorical_features_idx = list(map(lambda x: x[0], self.features_metadata["categorical"].values()))
        self._num_categorical_features = len(self._categorical_features_idx)
        self._numerical_features_idx = list(self.features_metadata["numerical"].values())
        self._num_numerical_features = len(self._numerical_features_idx)
        self._num_features = self._num_numerical_features + self._num_categorical_features

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
        if self.concatenate_numerical_features:
            encoded_features = tf.TensorArray(size=self._num_features,
                                              dtype=tf.float32)
        else:
            encoded_features = tf.TensorArray(size=self._num_categorical_features,
                                              dtype=tf.float32)
        current_idx = 0
        for feature_idx in self._categorical_features_idx:
            feature = tf.expand_dims(inputs[:, feature_idx], 1)
            # Convert string input values to integer indices
            lookup = self.lookup_tables[current_idx]
            feature = tf.cast(lookup(feature), dtype=tf.float32)
            encoded_features = encoded_features.write(index=feature_idx if (self.concatenate_numerical_features
                                                                            and self.keep_features_order) else current_idx,
                                                      value=feature)
            current_idx += 1

        if self.concatenate_numerical_features:
            for i, feature_idx in enumerate(self._numerical_features_idx):
                feature = tf.cast(tf.expand_dims(inputs[:, feature_idx], 1), tf.float32)
                encoded_features = encoded_features.write(index=feature_idx if self.keep_features_order else current_idx,
                                                          value=feature)
                current_idx += 1

        encoded_features = tf.squeeze(encoded_features.stack())
        encoded_features = tf.transpose(encoded_features)

        if self.concatenate_numerical_features:
            encoded_features.set_shape((None, self._num_features))
        else:
            encoded_features.set_shape((None, self._num_categorical_features))
        return encoded_features

    def get_config(self):
        config = super().get_config()
        new_config = {'features_metadata': self.features_metadata,
                      'concatenate_numerical_features': self.concatenate_numerical_features,
                      'keep_features_order': self.keep_features_order}
        config.update(new_config)
        return config
