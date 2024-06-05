from jax import numpy as jnp
import jax
from typing_extensions import NamedTuple

class MLP(NamedTuple):
  '''
  whatever parameters you need - usually the up projection and down projection
  '''
  d_model: int
  d_ff: int
  layers: list

def init_mlp(prng_key: jax.Array, d_model: int, d_ff: int):
  '''
  initialize the parameters and return an instance of MLP
  '''
  #initialize layers
  initializer = jax.nn.initializers.normal(0.01)
  layers = [[initializer(prng_key, (d_model,d_ff), jnp.float32), initializer(prng_key, (d_ff,), jnp.float32)], [initializer(prng_key, (d_ff, d_model), jnp.float32), initializer(prng_key, (d_model,), jnp.float32)]]

  #return instance
  return MLP(d_model, d_ff, layers)

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

def test_mlp():
  '''
  Create a random tensor for sequence
  Create the MLP using init_mlp
  Pass the sequence through the MLP using forward_mlp and capture the output
  Assert that the output if of the correct shape
  '''
  #params
  prng_key = jax.random.PRNGKey(0)
  d_model = 512
  d_ff = 2048
  batch_size = 32
  sequence_length = 16

  #create seq
  initializer = jax.nn.initializers.normal(0.01)
  seq = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

  #mlp transformation
  mlp = init_mlp(prng_key, d_model, d_ff)
  output = forward_mlp(mlp, seq)

  #compare outputs
  print('input shape: ')
  print(seq.shape)
  print('output shape: ')
  print(output.shape)

def main():
  return test_mlp()

main()