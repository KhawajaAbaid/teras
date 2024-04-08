import jax.numpy as jnp


def norm(x, ord, axis, keepdims):
    return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)