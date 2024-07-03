import mx
import logits_weights
import jax
import jax.numpy as jnp

def logits_weights_lookup(log_weights: logits_weights.LogitsWeights, seq: jax.Array):
    #log_weights_table = mx.quantize(log_weights.table, jnp.float8_e4m3fn)
    seq = mx.quantize(seq, jnp.float8_e4m3fn)
    return mx.mx_matmul(seq, mx.quantize(log_weights.table, jnp.float8_e4m3fn))