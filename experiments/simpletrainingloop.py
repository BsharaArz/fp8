import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple
import optax

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

#SOFTMAX AND CROSS ENTROPY

def softmax_cross_entropy(logits: jax.Array, labels: jax.Array):
  '''
  logits/labels has shape (batch_size, sequence_length, d_model)
  logits from training, labels from validation
  '''
  return optax.losses.softmax_cross_entropy(logits, labels)

#EMBEDDING

class Embedding(NamedTuple):
  table: jax.Array

def create_table(prng_key: jax.Array, d_model: int, vocab_size: int):
  #create table
  initializer = jax.nn.initializers.normal(0.01)
  table = initializer(prng_key, (vocab_size, d_model), jnp.float32)
  return Embedding(table)

'''
Embedding Lookup
input: array of int representing batch of sequences
input shape = (batch_size, sequence_length)

output: embedding from embedding table
output shape = (batch_size, sequence_length, d_model)
'''
def embedding_lookup(emb: Embedding, seq: jax.Array):
  #lookup seq from table
  embedded = jnp.take(emb.table, seq, axis=0)
  return embedded

#BACKWARD PASS

def calc_loss(params: MLP, input: jax.Array, target: jax.Array):
  #pass through forward, calculate loss compared w/ target
  logits = forward_mlp(params, input)
  loss = softmax_cross_entropy(logits, target)
  return loss.mean()

calc_grad = jax.grad(calc_loss, argnums=0, allow_int=True)

def training_loop(data, prng_key, num_epochs: int, d_model: int, d_ff: int, batch_size: int, sequence_length: int, num_batches: int, vocab_size: int, learning_rate):
  #initialize mlp
  mlp = init_mlp(prng_key, d_model, d_ff)

  target_mlp = init_mlp(prng_key, d_model, d_ff)

  #initialize embedding table
  table = create_table(prng_key, d_model, vocab_size)

  #initialize optimizer
  optimizer = optax.adamw(learning_rate)
  opt_state = optimizer.init(mlp)

  initializer = jax.nn.initializers.normal(0.01)
  for e in range(num_epochs):
    for b in range(num_batches):
      #extract seq and target from data
      seq = jax.random.randint(prng_key, (batch_size, sequence_length), 0, vocab_size) #(would be tokenized from data)

      #embed
      embeddedSeq = embedding_lookup(table, seq)

      #generate target
      target = forward_mlp(target_mlp, embeddedSeq)

      #backward pass
      loss = calc_loss(mlp, embeddedSeq, target)
      grad = calc_grad(mlp, embeddedSeq, target)

      updates, opt_state = optimizer.update(grad, opt_state, mlp)
      mlp = optax.apply_updates(mlp, updates)
      if b == 0:
        print(f"Epoch {e}, Loss: {loss}")

  return mlp

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