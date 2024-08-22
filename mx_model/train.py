from typing_extensions import NamedTuple
import jax
import optax
import data_batching
from jax import numpy as jnp
import functools
from matplotlib.pylab import plt
from tqdm import tqdm
import mx_llama
import mx

'''
SET GLOBAL VARIABLE CUSTOM MATMUL/SCALING TECHNIQUE
Options:
 - jnp.matmul
 - mx.tensor_matmul
 - mx.block_matmul
 - mx.kernel_matmul
'''
custom_matmul = jnp.matmul

#forward pass
@functools.partial(jax.jit, static_argnames=['num_heads']) 
def forward(llam, seq, num_heads, drop, prng_key, label):
  logits = mx_llama.forward_llama(llam, seq, num_heads, drop, prng_key).astype(jnp.float32)
  loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, label)
  return loss.mean()

#fwd-bwd grad
fwd_bwd = jax.grad(forward, argnums=0, allow_int=True)

#training step 
@functools.partial(jax.jit, static_argnames=['optimizer', 'num_heads']) 
def step_fn(llam, optimizer, opt_state, seq, num_heads, drop, prng_key, label):
    grad = fwd_bwd(llam, seq, num_heads, drop, prng_key, label)
    updates, opt_state = optimizer.update(grad, opt_state, llam)
    llam = optax.apply_updates(llam, updates)
    return llam, opt_state

train_loss_dict = {}
val_loss_dict = {}

#TRAIN
def train(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size, learning_rate, num_epochs, num_heads, drop, file_name):
    #tokenize file
    tokenized_file = data_batching.process_file(file_name)

    #initialize llama
    llam = mx_llama.init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size)
    
    #initialize optimizer
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate, 0.9, 0.95))
    opt_state = optimizer.init(llam)

    step = 0
    for e in range(num_epochs):
        print(f"Epoch {e}")

        #create batches
        batches = data_batching.create_batches(tokenized_file, batch_size, sequence_length)

        for seq, label in tqdm(batches):
            if step >= 2000:
                break
            #forward pass
            loss = forward(llam, jnp.array(seq), num_heads, drop, prng_key, jnp.array(label))

            #store loss
            train_loss_dict[step] = loss

            #optimizer
            llam, opt_state = step_fn(llam, optimizer, opt_state, jnp.array(seq), num_heads, drop, prng_key, jnp.array(label))

            step += 1
            print(f"Epoch {e} Loss {loss}")

    return llam

#VALIDATE
def validate(llam, valid_file_name, num_heads, drop, prng_key, batch_size, sequence_length):
    #tokenize
    print("tokenizing file")
    tokenized_file = data_batching.process_file_valid(valid_file_name)

    #steps    
    step = 0
    for seq, label in data_batching.create_batches(tokenized_file, batch_size, sequence_length):
        if step >= 2000:
            break
        #calculate loss
        loss = forward(llam, jnp.array(seq), num_heads, drop, prng_key, jnp.array(label))
        #store loss
        val_loss_dict[step] = loss

        step += 1
        print(f"Validation Loss {loss}")

def plot_loss():
    #plot and label the training and validation loss values
    plt.plot(train_loss_dict.keys(), train_loss_dict.values(), label='Training Loss')
    plt.plot(val_loss_dict.keys(), val_loss_dict.values(), label='Validation Loss')
    plt.title('Tensor-Level Quantization')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig('data/plot.png')
    #display
    plt.legend(loc='best')
    plt.show()

#----------------------TESTS------------------------

def test():
    #params
    prng_key = jax.random.PRNGKey(0)
    num_epochs = 2
    d_model = 128 #from paper
    d_ff = 512 #from paper
    batch_size = 1
    sequence_length = 512 #from paper
    vocab_size = 30000
    learning_rate = 0.001
    num_heads = 8
    num_blocks = 12 #from paper
    drop = 0.1
    train_file_name = "/Users/arzb/code_cleanup/fp8/experiments/data/TinyStories-valid.txt"
    valid_file_name = "data/TinyStories-valid.txt"
    
    llam = train(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size, learning_rate, num_epochs, num_heads, drop, train_file_name)

    validate(llam, valid_file_name, num_heads, drop, prng_key, batch_size, sequence_length)
    
    plot_loss()

def main():
    test()

if __name__ == "__main__":
    main()