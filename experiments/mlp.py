from jax import numpy as jnp
import jax
from typing_extensions import NamedTuple

#MLP LAYER

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
  initializer = jax.nn.initializers.normal(0.01)
  layers = [[initializer(prng_key, (d_model,d_ff)), initializer(prng_key, (d_ff,))], [initializer(prng_key, (d_ff, d_model)), initializer(prng_key, (d_model,))]]

  #return instance
  return MLP(layers)

def forward_mlp(params: MLP, seq: jax.Array):
  '''
  seq is Sequence - input to the MLP block.
  seq is of shape (batch_size, sequence_length, d_model
  Do the necessary matrix multiplications and return the transformer sequence
  '''
  #wx+b computations
  activations = seq
  for w, b in params.layers:
    activations = jax.nn.relu(jnp.dot(activations, w) + b)

  return activations

if name == "main":
  main()