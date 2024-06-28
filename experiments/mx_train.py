import llama
import optax
import jax
import functools
import mx_llama
import jax.numpy as jnp

#forward pass
#@functools.partial(jax.jit, static_argnames=['num_heads']) 
def forward(llam, seq, num_heads, drop, prng_key, label):
  logits = mx_llama.forward_llama(llam, seq, num_heads, drop, prng_key) # logits (batch, sequence_len, d_vocab)
  loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, label)
  return loss.mean()

#fwd-bwd grad
fwd_bwd = jax.grad(forward, argnums=0, allow_int = True)

def test():
    #params
    prng_key = jax.random.PRNGKey(0)
    num_epochs = 2
    d_model = 256 #from paper
    d_ff = 512 #from paper
    batch_size = 32
    sequence_length = 512 #from paper
    vocab_size = 30000
    learning_rate = 0.0005
    num_heads = 8
    num_blocks = 12 #from paper
    drop = 0.5


    fp32_llama = llama.init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size)
    fp8_llama = mx_llama.init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size)

    #initialize seq
    initializer = jax.nn.initializers.normal(0.01)
    seq = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)
    seq2 = jax.random.randint(prng_key, (batch_size, sequence_length), 0, d_model)
    print("FP32 loss:")
    print(forward(fp32_llama, seq, num_heads, 0, prng_key, seq2))
    #print(fwd_bwd(fp32_llama, seq, num_heads, 0, prng_key, seq2).tran.blocks[0])

    print("FP8 loss:")
    print(forward(fp8_llama, seq, num_heads, drop, prng_key, seq2))
    #print(fwd_bwd(fp8_llama, seq, num_heads, drop, prng_key, seq2).tran.blocks[0])

def main():
    test()

test()

# if name == "main":
#     main()