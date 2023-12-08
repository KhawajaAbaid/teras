import keras
from keras import ops, random


# ================================= GLU =======================================
@keras.saving.register_keras_serializable(package="teras.activations")
def glu(logits: keras.KerasTensor) -> keras.KerasTensor:
    """
    Generalized linear unit nonlinear activation.

    logits: ``Tensor``,
        Input tensor of logits.
    """
    x, gates = ops.split(logits,
                         indices_or_sections=2,
                         axis=-1)
    return x * ops.sigmoid(gates)


# ================================= GEGLU =======================================
@keras.saving.register_keras_serializable(package="teras.activations")
def geglu(logits: keras.KerasTensor) -> keras.KerasTensor:
    """
    GeGLU is an activation function which is a variant of GLU

    logits: ``Tensor``,
        Input tensor of logits.
    """
    x, gates = ops.split(logits,
                         indices_or_sections=2,
                         axis=-1)
    return x * ops.gelu(gates)


# ================================= SPARSEMAX =======================================
# The function below is copied from TensorFlow Addons with slight modifications.
# And reason we copied it here because TensorFlow Addons has ended development.
# RIP o7
@keras.saving.register_keras_serializable(package="teras.activations")
def sparsemax(logits: keras.KerasTensor, axis: int = -1) -> keras.KerasTensor:
    """
    Sparsemax activation function.

    Referencne(s):
        https://arxiv.org/abs/1602.02068.

    Usage:

    >>> x = keras.ops.array([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
    >>> teras.activations.sparsemax(x)
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0., 0., 1.],
           [0., 0., 1.]], dtype=float32)>

    Args:
        logits: ``Tensor``.
        axis: ``int``, axis along which the sparsemax operation is applied.
    Returns:
        A `Tensor`, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    # We need its original shape for shape inference.
    shape = ops.shape(logits)
    rank = len(shape)
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op =(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, ops.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, ops.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return ops.transpose(
        logits,
        ops.concatenate(
            [
                ops.arange(dim_index),
                [last_index],
                ops.arange.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = ops.shape(logits)
    obs = ops.prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = ops.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = ops.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = ops.cumsum(z_sorted, axis=-1)
    k = ops.arange(1, ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = ops.sum(ops.cast(z_check, "int32"), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = ops.maximum(k_z, 1)
    indices = ops.stack([ops.arange(0, obs), ops.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = ops.take(z_cumsum, indices)
    tau_z = (tau_sum - 1) / ops.cast(k_z, logits.dtype)

    # calculate p
    p = ops.maximum(ops.cast(0, logits.dtype), z - ops.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = ops.where(
        ops.expand_dims(
            ops.logical_or(ops.equal(k_z, 0), ops.isnan(z_cumsum[:, -1])),
            axis=-1,
        ),
        ops.full([obs, dims], ops.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = ops.reshape(p_safe, shape_op)
    return p_safe


# ================================= GUMBLE SOFTMAX =======================================
@keras.saving.register_keras_serializable(package="teras.activations")
def gumbel_softmax(logits: keras.KerasTensor,
                   temperature: float = 0.2,
                   hard: bool = False) -> keras.KerasTensor:
    """
    Implementation of the Gumbel Softmax activation function
    proposed by Eric Jang et al. in the paper
    Categorical Reparameterization with Gumbel-Softmax

    Reference(s):
        https://arxiv.org/abs/1611.01144

    Args:
        logits: ``Tensor`` or ``ndarray``
            Input tensor of logits.

        temperature: ``float``, default 0.2,
            Controls the sharpness or smoothness of the resulting probability distribution.
            A higher temperature value leads to a smoother and more uniform probability distribution.
            Conversely, a lower temperature value makes the distribution concentrated around
            the category with the highest probability.

        hard: ``bool``, default False,
            Whether to return soft probabilities or hard one hot vectors.
    """
    u = random.uniform(ops.shape(logits),
                       minval=0,
                       maxval=1)
    gumbels = -ops.log(-ops.log(u))
    perturbed_logits = (logits + gumbels) / temperature
    probabilities = ops.nn.softmax(perturbed_logits)
    if hard:
        one_hot_labels = ops.one_hot(ops.argmax(probabilities, axis=-1),
                                     ops.shape(logits)[-1])
        return one_hot_labels
    return probabilities
