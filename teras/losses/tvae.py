import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses


class TvaeElboLoss(layers.Layer):
    """
    Evidence Lower Bound (ELBO) Loss [1] adapted for
    TVAE architecture [2] proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        [1]: https://arxiv.org/abs/1312.6114
        [2]: https://arxiv.org/abs/1907.00503

    Args:
        features_meta_data: A dictionary containing meta data for all
            features in the input.
            Usually available from data_transformer.features_meta_data
    """
    def __init__(self,
                 features_meta_data,
                 **kwargs):
        super().__init__(**kwargs)
        self.features_meta_data = features_meta_data
        self.num_valid_clusters_all = self.features_meta_data["continuous"]["num_valid_clusters_all"]
        self.num_continuous_features = len(self.num_valid_clusters_all)
        self.features_relative_indices_all = self.features_meta_data["relative_indices_all"]
        self.num_categories_all = self.features_meta_data["categorical"]["num_categories_all"]
        self.cross_entropy = losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=losses.Reduction.SUM)
        self.inputs_dim = None

    def build(self, input_shape):
        self.inputs_dim = tf.cast(input_shape[1], tf.float32)

    def call(self,
             x_real,
             x_generated=None,
             sigmas=None,
             mean=None,
             log_var=None,
             loss_factor=None):
        loss = []
        cont_i = 0 # continuous index
        cat_i = 0 # categorical index
        for i, relative_index in enumerate(self.features_relative_indices_all):
            # the first k features are continuous
            if i < self.num_continuous_features:
                # each continuous feature is of the form
                # [alpha, beta1, beta2...beta(n)] where n is the number of clusters

                # calculate alpha loss
                std = sigmas[relative_index]
                eq = x_real[:, relative_index] - tf.nn.tanh(x_generated[:, relative_index])
                loss_temp = tf.reduce_sum((eq ** 2 / 2 / (std ** 2)))
                loss.append(loss_temp)
                loss_temp = tf.math.log(std) * tf.cast(tf.shape(x_real)[0], dtype=tf.float32)
                loss.append(loss_temp)

                # calculate betas loss
                num_clusters = self.num_valid_clusters_all[cont_i]
                logits = x_generated[:, relative_index+1: relative_index + 1 + num_clusters]
                labels = tf.argmax(x_real[:, relative_index+1: relative_index+1+num_clusters], axis=-1)
                cross_entropy_loss = self.cross_entropy(y_pred=logits, y_true=labels)
                loss.append(cross_entropy_loss)
                cont_i += 1
            else:
                num_categories = self.num_categories_all[cat_i]
                logits = x_generated[:, relative_index: relative_index + num_categories]
                labels = tf.argmax(x_real[:, relative_index: relative_index + num_categories], axis=-1)
                cross_entropy_loss = self.cross_entropy(y_pred=logits, y_true=labels)
                loss.append(cross_entropy_loss)
                cat_i += 1
        KLD = -0.5 * tf.reduce_sum(1 + log_var - mean**2 - tf.exp(log_var))
        loss_1 = tf.reduce_sum(loss) * loss_factor / self.inputs_dim
        loss_2 = KLD / self.inputs_dim
        final_loss = loss_1 + loss_2
        return final_loss
