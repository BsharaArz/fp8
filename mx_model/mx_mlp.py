from jax import numpy as jnp
import jax
from typing_extensions import NamedTuple

import mx
import train

class MLP(NamedTuple):
  '''
  whatever parameters you need - usually the up projection and down projection
  '''
  layers: list

def init_mlp(prng_key: jax.Array, d_model: int, d_ff: int):
  '''
  initialize the parameters and return an instance of MLP
  '''
  #initialize layers
  cpu_0 = jax.devices('cpu')[0]
  with jax.default_device(cpu_0):
    initializer = jax.nn.initializers.normal(0.01)
    layers = [[initializer(prng_key, (d_model,d_ff)), initializer(prng_key, (d_ff,))], [initializer(prng_key, (d_ff, d_model)), initializer(prng_key, (d_model,))]]

  #return instance
  return MLP(layers)

def forward_mlp(params: MLP, seq: jax.Array):
  '''
  seq is Sequence - input to the MLP block - MX FORMAT
  seq is of shape (batch_size, sequence_length, d_model
  Do the necessary matrix multiplications and return the transformer sequence
  '''
  #wx+b computations
  activations = seq
  for w, b in params.layers:#[:-1]:
    activations = jax.nn.relu(train.custom_matmul(activations, w) + b)

  return activations

#----------------------TESTS------------------------

def test():
    #initialize params
    prng_key = jax.random.PRNGKey(0)
    d_model = 512
    d_ff = 1024
    batch_size = 32
    sequence_length = 16

    #init mlp
    params = init_mlp(prng_key, d_model, d_ff)

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

if __name__ == "__main__":
    main()