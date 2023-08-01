import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.dnfnet")
class DNFNetLocalization(keras.layers.Layer):
    """
    DNFNetLocalization layer based on the localization component
    proposed by Liran Katzir et al.
    in the paper Net-DNF: Effective Deep Modeling of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_formulas: ``int``, default 256,
            Number of DNF formulas to use.
            Each DNF formula is analogous to a tree in tree based ensembles.

        temperature: ``float``, default 2.0,
            Temperature value to use.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_formulas: int = 256,
                 temperature: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_formulas = num_formulas
        self.temperature = temperature
        self.softmax = keras.layers.Softmax()
        self.mu = None
        self.sigma = None

    def build(self, input_shape):
        input_dim = input_shape[1]
        mu_initializer = keras.initializers.random_normal()
        self.mu = tf.Variable(initial_value=mu_initializer(shape=(self.num_formulas, input_dim)),
                              shape=(self.num_formulas, input_dim),
                              name='exp_mu')
        sigma_initializer = keras.initializers.random_normal()
        self.sigma = tf.Variable(initial_value=sigma_initializer(shape=(1, self.num_formulas, input_dim)),
                                 shape=(1, self.num_formulas, input_dim),
                                 name="exp_sigma")
        self.temperature = tf.Variable(name='temperature',
                                       initial_value=tf.constant(value=self.temperature),
                                       dtype=tf.float32)

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.mu, axis=0)
        loc = tf.exp(-1 * tf.norm(tf.multiply(diff, self.sigma), axis=-1))
        loc = self.softmax(tf.sigmoid(self.temperature) * loc)
        return loc

    def get_config(self):
        config = super().get_config()
        config.update({'num_formulas': self.num_formulas,
                       'temperature': self.temperature}
                      )
        return config
