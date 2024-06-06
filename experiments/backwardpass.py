import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple
import optax

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

#forward pass
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

#softmax cross entropy calculation
def calculate(logits: jax.Array, labels: jax.Array):
  '''
  logits/labels has shape (batch_size, sequence_length, d_model)
  logits from training, labels from validation
  '''
  return optax.losses.softmax_cross_entropy(logits, labels)

def calc_loss(params: MLP, input: jax.Array, target: jax.Array):
  #pass through forward, calculate loss compared w/ target
  logits = forward_mlp(params, input)
  loss = calculate(logits, target)
  return loss.mean()

calc_grad = jax.grad(calc_loss, argnums=0, allow_int=True)

def test():
  #params
  prng_key = jax.random.PRNGKey(0)
  d_model = 512
  d_ff = 2048
  batch_size = 32
  sequence_length = 16

  #initialize mlp
  mlp = init_mlp(prng_key, d_model, d_ff)

  #create input and target
  initializer = jax.nn.initializers.normal(0.01)
  input = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)
  target = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

  #calc grad
  gradient = calc_grad(mlp, input, target)
  print(gradient)

def main():
  test()

if name == "main":
  main()