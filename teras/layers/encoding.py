import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LabelEncoding(layers.Layer):
    """
    Standalone Encoding layer
    Essentially it acts as a Label encoder.
    Useful in cases like SAINT Pretraining where we want to compute the
    difference between original and reconstructed inputs and for cases
    when original inputs contain features with strings, we can simply
    call the encode method to convert them to their respective integer
    labels.

    Args:
        inputs: Inputs with string features that should be encoded
        concatenate_numerical_features: `bool`, default False,
            Whether to concatenate numerical features
            to the encoded categorical features.
        keep_features_order: `bool`, default True,
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
                 categorical_features_metadata: dict,
                 concatenate_numerical_features: bool = False,
                 keep_features_order: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_features_metadata = categorical_features_metadata
        self.concatenate_numerical_features = concatenate_numerical_features
        self.keep_features_order = keep_features_order

        # since we only have metadata for categorical values e.g. feature names and indices
        # we can still get indices and names for the numerical features --
        # Just see the magic that's about to happen. aabraca... whoops not that magic
        self._categorical_features_idx = set(map(lambda x: x[0], self.categorical_features_metadata.values()))
        self._categorical_features_names = set(self.categorical_features_metadata.keys())
        self._num_categorical_features = len(self.categorical_features_metadata)
        self._num_features = None
        self._numerical_features_names = None
        self._numerical_features_idx = None

        self.lookup_tables = self._get_lookup_tables()

        self._is_first_batch = True
        self._is_data_in_dict_format = False

    def _get_lookup_tables(self):
        """Lookup tables for each categorical feature"""
        lookup_tables = []
        for feature_name, (feature_idx, vocabulary) in self.categorical_features_metadata.items():
            # Lookup Table to convert string values to integer indices
            lookup = layers.StringLookup(vocabulary=vocabulary,
                                         mask_token=None,
                                         num_oov_indices=0,
                                         output_mode="int"
                                         )
            lookup_tables.append(lookup)
        return lookup_tables

    def call(self, inputs):
        if self._is_first_batch:
            if isinstance(inputs, dict):
                self._is_data_in_dict_format = True
                self._num_features = len(inputs)
                self._numerical_features_names = list(set(inputs.keys()) - self._categorical_features_names)
            else:
                self._num_features = tf.shape(inputs)[1]

            self._numerical_features_idx = set(range(self._num_features)) - self._categorical_features_idx
            self._is_first_batch = False

        if self.concatenate_numerical_features:

            encoded_features = tf.TensorArray(size=self._num_features,
                                              dtype=tf.float32)
        else:
            encoded_features = tf.TensorArray(size=self._num_categorical_features,
                                              dtype=tf.float32)

        current_idx = 0
        for feature_name, (feature_idx, _) in self.categorical_features_metadata.items():
            if self._is_data_in_dict_format:
                feature = tf.expand_dims(inputs[feature_name], 1)
            else:
                feature = tf.expand_dims(inputs[:, feature_idx], 1)
            # Convert string input values to integer indices
            lookup = self.lookup_tables[current_idx]
            feature = tf.cast(lookup(feature), dtype=tf.float32)
            encoded_features = encoded_features.write(index=feature_idx if self.keep_features_order else current_idx,
                                                      value=feature)
            current_idx += 1

        if self.concatenate_numerical_features:
            for i, feature_idx in enumerate(self._numerical_features_idx):
                if self._is_data_in_dict_format:
                    feature = tf.expand_dims(inputs[self._numerical_features_names[i]], 1)
                else:
                    feature = tf.expand_dims(inputs[:, feature_idx], 1)
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
        new_config = {'categorical_features_metadata': self.categorical_features_metadata,
                      'concatenate_numerical_features': self.concatenate_numerical_features,
                      'keep_features_order': self.keep_features_order}
        config.update(new_config)
        return config

