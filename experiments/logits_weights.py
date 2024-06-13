import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple

class Logits_Weights(NamedTuple):
    table: jax.Array

def init_logits_weights(prng_key: jax.Array, d_model: int, vocab_size: int):
    table = jax.random.randint(prng_key, (d_model, vocab_size), 0, vocab_size)
    return Logits_Weights(table)

def logits_weights_lookup(log_weights: Logits_Weights, seq: jax.Array):
    return jnp.take(log_weights.table, seq)