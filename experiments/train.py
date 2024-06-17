import llama
from typing_extensions import NamedTuple
import jax
import optax
import softmax_entropy
import data_batching
import jax.numpy as jnp

#forward pass
def forward(llam, seq, num_heads, drop, prng_key, label):
  logits = llama.forward_llama(llam, seq, num_heads, drop, prng_key) # logits (batch, sequence_len, d_vocab)
  loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, label)
  return loss.mean()

#fwd-bwd grad
fwd_bwd = jax.grad(forward, argnums=0, allow_int = True)

#training step 
def step_fn(llam, optimizer, opt_state, seq, num_heads, drop, prng_key, label):
    grad = fwd_bwd(llam, seq, num_heads, drop, prng_key, label)
    updates, opt_state = optimizer.update(grad, opt_state, llam)
    llam = optax.apply_updates(llam, updates)
    return llam, opt_state

def train(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size, learning_rate, num_epochs, num_heads, drop, file_name):
    #tokenize file
    tokenized_file = data_batching.process_file(file_name)

    #initialize llama
    llam = llama.init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size)

    #initialize optimizer
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(llam)
    for e in range(num_epochs):
        print(f"Epoch {e}")
        for seq, label in data_batching.create_batches(tokenized_file, batch_size, sequence_length):
            loss = forward(llam, jnp.array(seq), num_heads, drop, prng_key, jnp.array(label))
            print(f"Loss {loss}")
            llam, opt_state = step_fn(llam, optimizer, opt_state, jnp.array(seq), num_heads, drop, prng_key, jnp.array(label))

def test():
    #params
    prng_key = jax.random.PRNGKey(0)
    num_epochs = 10
    d_model = 512
    d_ff = 2048
    batch_size = 64
    sequence_length = 32
    vocab_size = 30000
    learning_rate = 0.00001
    num_heads = 8
    num_blocks = 12
    drop = 0.5
    file_name = "experiments/data/TinyStories-train.txt"

    train(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size, learning_rate, num_epochs, num_heads, drop, file_name)

def main():
    test()

if name == "main":
    main()