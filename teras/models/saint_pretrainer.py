from tensorflow import keras
from teras.layerflow.models.saint_pretrainer import SAINTPretrainer as _SAINTPretrainerLF
from teras.layers.saint.saint_reconstruction_head import SAINTReconstructionHead
from teras.layers.saint.saint_projection_head import SAINTProjectionHead
from teras.layers.regularization import MixUp, CutMix


@keras.saving.register_keras_serializable(package="keras.models")
class SAINTPretrainer(_SAINTPretrainerLF):
    """
    SAINTPretrainer model based on the pretraining architecture
    for the SAINT model proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        model: ``keras.Model``,
            An instance of the ``SAINT`` model that you want to pretrain.
            Note that, you should use the base ``SAINT`` model's instance,
            not ``SAINTClassifier`` or ``SAINTRegressor``.
            Using default API, you can import it as,
                >>> from teras.models import SAINT
            Using LayerFlow API, you can import it as,
                >>> from teras.layerflow.models import SAINT
                And REMEMBER to leave the ``head`` argment as None.

        cutmix_probs: ``float``, default 0.1,
            ``CutMix`` probability which is used in generation of mask
            that is used to mix samples together.

        mixup_alpha: ``float``, default 1.0,
            Alpha value for the ``MixUp`` layer, that is used for the
            Beta distribution to sample `lambda_`
            which is used to interpolate samples.

        temperature: ``float``, default 0.7,
            Temperature value used in the computation of the InfoNCE contrastive loss.

        lambda_: ``float``, default 10,
            Controls the weightage of denoising loss in the summation of denoising and
            contrastive loss.
    """
    def __init__(self,
                 model: keras.Model,
                 cutmix_probs: float = 0.3,
                 mixup_alpha: float = 1.0,
                 temperature: float = 0.7,
                 lambda_: float = 10.,
                 **kwargs):
        mixup = MixUp(alpha=mixup_alpha)
        cutmix = CutMix(probs=cutmix_probs)

        # For the computation of contrastive loss, we use projection heads.
        # Projection head hidden dimensions as calculated by the
        # official implementation
        projection_head_hidden_dim = 6 * model.embedding_dim * model.num_features // 5
        projection_head_output_dim = model.embedding_dim * model.num_features // 2

        projection_head_1 = SAINTProjectionHead(hidden_dim=projection_head_hidden_dim,
                                                output_dim=projection_head_output_dim,
                                                name="projection_head_for_original_data")

        projection_head_2 = SAINTProjectionHead(hidden_dim=projection_head_hidden_dim,
                                                output_dim=projection_head_output_dim,
                                                name="projection_head_for_augmented_data")

        reconstruction_head = SAINTReconstructionHead(features_metadata=model.features_metadata,
                                                      embedding_dim=model.embedding_dim)

        super().__init__(model=model,
                         mixup=mixup,
                         cutmix=cutmix,
                         projection_head_1=projection_head_1,
                         projection_head_2=projection_head_2,
                         reconstruction_head=reconstruction_head)
        self.model = model
        self.cutmix_probs = cutmix_probs
        self.mixup_alpha = mixup_alpha
        self.temperature = temperature
        self.lambda_ = lambda_

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'model': keras.layers.serialize(self.model),
                  'cutmix_probs': self.cutmix_probs,
                  'mixup_alpha': self.mixup_alpha,
                  'temperature': self.temperature,
                  'lambda_': self.lambda_,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        return cls(model, **config)
