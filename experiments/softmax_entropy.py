import jax
import jax.numpy as jnp
import optax

'''
Description:
input: shape (batch_size, sequence_length, d_model)
output: shape (batch_size, sequence_length)
use optax function that combines softmax and cross entropy
'''

def calculate(logits: jax.Array, labels: jax.Array):
  '''
  logits/labels have shape (batch_size, sequence_length, d_model)
  logits from training, labels from validation
  '''
  return optax.losses.softmax_cross_entropy(logits, labels)

def test():
  #params
  batch_size = 32
  sequence_length = 16
  d_model = 512

  #create logits, labels
  prng_key = jax.random.PRNGKey(0)
  initializer = jax.nn.initializers.normal(0.01)
  logits = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)
  labels = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

  #calculate softmax + cross entropy
  output = calculate(logits, labels)
  print(output)

  #check that shape = (batch_size, sequence_length)
  print(output.shape)

def main():
  test()

if name == "main":
  main()