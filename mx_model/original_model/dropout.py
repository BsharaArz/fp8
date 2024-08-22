import jax
import jax.numpy as jnp

def dropout_layer(x, dropout, key):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(x)
    mask = jax.random.uniform(key, x.shape) > dropout
    return jnp.asarray(mask, jnp.float32) * x / (1.0 - dropout)