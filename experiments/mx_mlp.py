from jax import numpy as jnp
import jax
from typing_extensions import NamedTuple

import mx
import mlp

#MLP object/init same as fp32 (on mlp.py)

def forward_mlp(params: mlp.MLP, seq: mx.MX):
  '''
  seq is Sequence - input to the MLP block - MX FORMAT
  seq is of shape (batch_size, sequence_length, d_model
  Do the necessary matrix multiplications and return the transformer sequence
  '''
  #wx+b computations
  activations = seq
  for w, b in params.layers[:-1]:
    activations = mx.quantize(jax.nn.relu(mx.mx_matmul(activations, mx.quantize(w, jnp.float8_e4m3fn)) + b), jnp.float8_e4m3fn)

  w, b = params.layers[-1]
  activations = jax.nn.relu(mx.mx_matmul(activations, mx.quantize(w, jnp.float8_e4m3fn)) + b)

  return activations

def test():
    #initialize params
    prng_key = jax.random.PRNGKey(0)
    d_model = 512
    d_ff = 1024
    batch_size = 32
    sequence_length = 16

    #init mlp
    params = mlp.init_mlp(prng_key, d_model, d_ff)

    #initialize seq
    initializer = jax.nn.initializers.normal(0.01)
    seq = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

    #fp8 seq
    quantized_seq = mx.quantize(seq, jnp.float8_e4m3fn)

    #compare fp32, fp8 outputs
    print("FP32:")
    print(mlp.forward_mlp(params, seq)[0][0])

    print("FP8:")
    print(forward_mlp(params, quantized_seq)[0][0])

def main():
    test()

if name == "main":
  main()