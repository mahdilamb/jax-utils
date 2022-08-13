import jax
from jax import lax
import jax.numpy as jnp


def binary_crossentropy(target, output, from_logits=False):
    def not_from_logits(output):
        output = jnp.clip(output, 1e-7, 1 - 1e-7)
        return jnp.log(output / (1 - output))
    output = lax.cond(
        from_logits,
        lambda out: out,
        not_from_logits,
        output
    )
    return (target * -jnp.log(jax.nn.sigmoid(output)) +
            (1 - target) * -jnp.log(1 - jax.nn.sigmoid(output)))


def categorical_crossentropy(target: jnp.ndarray, output: jnp.ndarray, from_logits: bool = False):
    """Calculate the categorical cross entropy. Code adapated from old Keras Numpy backend

    Args:
        target (jnp.ndarray): The target labels
        output (jnp.ndarray): The model predictions
        from_logits (bool, optional): Whether the predictions are from logits. Defaults to False.

    Returns:
        _type_: _description_
    """
    output = lax.cond(
        from_logits,
        lambda out: jax.nn.softmax(out),
        lambda out: out / out.sum(axis=-1, keepdims=True),
        output
    )
    output = jnp.clip(output, 1e-7, 1 - 1e-7)
    eps = jnp.finfo(float).eps
    return -jnp.sum(target * jnp.log(output + eps))


def sparse_categorical_crossentropy(target, output, from_logits=False):
    return categorical_crossentropy(jax.nn.one_hot(target, output.shape[-1]), output=output, from_logits=from_logits)
