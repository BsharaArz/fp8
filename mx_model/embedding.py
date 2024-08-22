import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple

'''
Embedding Lookup
input: array of int representing batch of sequences
input shape = (batch_size, sequence_length)

output: embedding from embedding table
output shape = (batch_size, sequence_length, d_model)
'''

class Embedding(NamedTuple):
  table: jax.Array

def create_table(prng_key: jax.Array, d_model: int, vocab_size: int):
  #create table
  cpu_0 = jax.devices('cpu')[0]
  with jax.default_device(cpu_0):
    initializer = jax.nn.initializers.normal(0.01)
    table = initializer(prng_key, (vocab_size, d_model), jnp.float32)
  return Embedding(table)


def embedding_lookup(emb: Embedding, seq: jax.Array):
  #lookup seq from table
  embedded = jnp.take(emb.table, seq, axis=0)
  return embedded

#----------------------TESTS------------------------

def test():
  #initialize params
  batch_size = 32
  sequence_length = 16
  d_model = 512
  vocab_size = 2000
  prng_key = jax.random.PRNGKey(0)

  #create seq - only integers
  seq = jax.random.randint(prng_key, (batch_size, sequence_length), 0, vocab_size)

  #create table
  table = create_table(prng_key, d_model, vocab_size)
  output = embedding_lookup(table, seq)

  #check shape = (batch_size, sequence_length, d_model)
  print("shape:")
  print(output.shape)

def main():
  test()

if __name__ == "__main__":
    main()