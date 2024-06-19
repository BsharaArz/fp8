import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple

class Logits_Weights(NamedTuple):
    table: jax.Array

def init_logits_weights(prng_key: jax.Array, d_model: int, vocab_size: int):
    #create table
    cpu_0 = jax.devices('cpu')[0]
    with jax.default_device(cpu_0):
        initializer = jax.nn.initializers.normal(0.01)
        table = initializer(prng_key, (d_model, vocab_size), jnp.float32)

    return Logits_Weights(table)

def logits_weights_lookup(log_weights: Logits_Weights, seq: jax.Array):
    return jnp.matmul(seq, log_weights.table)