import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple
import optax
import mlp
import backwardpass
import embedding

def training_loop(data, prng_key, num_epochs: int, d_model: int, d_ff: int, batch_size: int, sequence_length: int, num_batches: int, vocab_size: int, learning_rate):
  #initialize mlp
  mlp_layer = mlp.init_mlp(prng_key, d_model, d_ff)

  target_mlp = mlp.init_mlp(prng_key, d_model, d_ff)

  #initialize embedding table
  table = embedding.create_table(prng_key, d_model, vocab_size)

  #initialize optimizer
  optimizer = optax.adamw(learning_rate)
  opt_state = optimizer.init(mlp_layer)

  for e in range(num_epochs):
    for b in range(num_batches):
      #extract seq and target from data
      seq = jax.random.randint(prng_key, (batch_size, sequence_length), 0, vocab_size) #(would be tokenized from data)

      #embed
      embeddedSeq = embedding.embedding_lookup(table, seq)

      #generate target
      target = mlp.forward_mlp(target_mlp, embeddedSeq)

      #backward pass
      loss = backwardpass.calc_loss(mlp_layer, embeddedSeq, target)
      grad = backwardpass.calc_grad(mlp_layer, embeddedSeq, target)

      updates, opt_state = optimizer.update(grad, opt_state, mlp_layer)
      mlp_layer = optax.apply_updates(mlp_layer, updates)
      if b == 0:
        print(f"Epoch {e}, Loss: {loss}")

  return mlp_layer

def test():
  #params
  data = None #when have data will modify
  prng_key = jax.random.PRNGKey(0)
  num_epochs = 10
  d_model = 512
  d_ff = 2048
  batch_size = 32
  sequence_length = 16
  num_batches = 10
  vocab_size = 2000
  learning_rate = 0.001

  training_loop(data, prng_key, num_epochs, d_model, d_ff, batch_size, sequence_length, num_batches, vocab_size, learning_rate)

def main():
  test()

if name == "main":
  main()