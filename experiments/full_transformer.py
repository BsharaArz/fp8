import jax
import jax.numpy as jnp
import optax

import attention
import backwardpass
import data_batching
import embeddingok
import mlp
import softmax_entropy

def model(file_name, batch_size, sequence_length, d_model, vocab_size, prng_key, num_epochs, d_ff, learning_rate, num_heads, drop):
  #process file
  file_tokenized = data_batching.process_file(file_name)

  #data batching
  num_batches = len(file_tokenized)//batch_size + (len(file_tokenized) % batch_size != 0)

  #initialize mlp and attention
  mlp_layer = mlp.init_mlp(prng_key, d_model, d_ff)
  attn_layer = attention.init_attention(prng_key, batch_size, sequence_length, d_model)

  #initialize embedding table
  table = embeddingok.create_table(prng_key, d_model, vocab_size)

  #initialize optimizer
  optimizer = optax.adamw(learning_rate)
  params = list(mlp_layer) + list(attn_layer)
  opt_state = optimizer.init(params)

  #initializer for qvk
  initializer = jax.nn.initializers.normal(0.01)

  for e in range(num_epochs):
    print(f"Epoch {e}")
    count = 0
    for batch, target in data_batching.create_batches(file_tokenized, batch_size, sequence_length):
      #embed
      embeddedSeq = embeddingok.embedding_lookup(table, jnp.array(batch))
      embeddedTarget = embeddingok.embedding_lookup(table, jnp.array(target))

      #forward + backward pass
      loss = backwardpass.calc_loss(params, embeddedSeq, embeddedTarget, num_heads, drop, prng_key)
      grad = backwardpass.calc_grad(params, embeddedSeq, embeddedTarget, num_heads, drop, prng_key)

      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      mlp_layer = mlp.MLP(params[0])
      attn_layer = attention.Attention(params[1], params[2], params[3])
      print(f"Loss: {loss}")
      count += 1

      if count == 10:
        break
        #used for testing (train only 10 batches per epoch)
  return params


def test():
  #params
  data = None #when have data will modify
  prng_key = jax.random.PRNGKey(0)
  num_epochs = 10
  d_model = 512
  d_ff = 2048
  batch_size = 64
  sequence_length = 32
  vocab_size = 30000
  learning_rate = 0.00001
  num_heads = 8
  drop = 0.5
  file_name = "TinyStories-train.txt"

  model(file_name, batch_size, sequence_length, d_model, vocab_size, prng_key, num_epochs, d_ff, learning_rate, num_heads, drop)

def main():
  test()

if name == "main": 
  main()