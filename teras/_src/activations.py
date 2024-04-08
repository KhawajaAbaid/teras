from keras import ops, random
from teras._src.api_export import teras_export


# =========================== GLU ===========================
@teras_export("teras.activations.glu")
def glu(logits, axis: int = -1):
    """
    Generalized linear unit nonlinear activation.

    Args:
        logits: `Tensor`, tensor of logits.
        axis: `int`, axis along which to apply glu activation.
            Defaults to -1.
    """
    x, gates = ops.split(logits,
                         indices_or_sections=2,
                         axis=axis)
    return x * ops.sigmoid(gates)


# =========================== GEGLU ===========================
@teras_export("teras.activations.geglu")
def geglu(logits, axis: int = -1):
    """
    GeGLU is an activation function which is a variant of GLU

    Args:
        logits: `Tensor`, tensor of logits.
        axis: `int`, axis along which to apply geglu activation.
            Defaults to -1.
    """
    x, gates = ops.split(logits,
                         indices_or_sections=2,
                         axis=axis)
    return x * ops.gelu(gates)


# =========================== SPARSEMAX ===========================
@teras_export("teras.activations.sparsemax")
def sparsemax(logits, axis: int = -1):
    """
    Sparsemax activation function as proposed by T. Martins et al. in
    the paper, "From Softmax to Sparsemax: A Sparse Model of Attention
    and Multi-Label Classification"

    Reference(s):
        https://arxiv.org/abs/1602.02068

    Args:
        logits: `Tensor`, tensor of logits.
        axis: `int`, axis along which to apply the sparsemax activation.
            Defaults to -1.
    """
    K = ops.shape(logits)[-1]
    idx = ops.expand_dims(ops.arange(1, K+1, dtype=logits.dtype), 0)
    z_sorted, _ = ops.top_k(logits, k=K)
    z_cumsum = ops.cumsum(z_sorted, axis=axis, dtype=logits.dtype)
    kz = ops.sum(1 + (idx * z_sorted) > z_cumsum, axis=axis, keepdims=True)
    # subtract 1 from kz to bring indices in range [0, K),
    # instead of (0, K]
    selective_cumsum = ops.take_along_axis(z_cumsum, indices=kz - 1,
                                           axis=axis)

    threshold = (selective_cumsum - 1) / ops.cast(kz, selective_cumsum.dtype)
    logits_sub_threshold = logits - threshold
    p = ops.relu(logits_sub_threshold)
    return p


# =========================== Gumbel Softmax ===========================
@teras_export("teras.activations.gumbel_softmax")
def gumbel_softmax(logits,
                   temperature: float = 0.2,
                   hard: bool = False,
                   seed: int = None):
    """
    Implementation of the Gumbel Softmax activation function
    proposed by Eric Jang et al. in the paper
    Categorical Reparameterization with Gumbel-Softmax

    Reference(s):
        https://arxiv.org/abs/1611.01144

    Args:
        logits: `Tensor`
            Input tensor of logits.
        temperature: `float`, default 0.2,
            Controls the sharpness or smoothness of the resulting
            probability distribution. A higher temperature value leads
            to a smoother and more uniform probability distribution.
            Conversely, a lower temperature value makes the distribution
            concentrated around the category with the highest probability.
        hard: `bool`, default `False`,
            Whether to return soft probabilities or hard one hot vectors.
        seed: int, seed to use for random sampling.
    """
    u = random.uniform(ops.shape(logits),
                       minval=0,
                       maxval=1,
                       seed=seed)
    gumbels = -ops.log(-ops.log(u))
    perturbed_logits = (logits + gumbels) / temperature
    probabilities = ops.nn.softmax(perturbed_logits)
    if hard:
        one_hot_labels = ops.one_hot(ops.argmax(probabilities, axis=-1),
                                     ops.shape(logits)[-1])
        return one_hot_labels
    return probabilities
