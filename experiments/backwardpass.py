import jax 
import jax.numpy as jnp
import mlp
import softmax_entropy
import attention
import dropout


def calc_loss(params: list, seq: jax.Array, target: jax.Array, num_heads, drop, prng_key):
  #separate mlp, attention params
  mlp_params = mlp.MLP(params[0])
  attn_params = attention.Attention(params[1], params[2], params[3])

  #forward attention
  attn, _ = attention.forward_attention(attn_params, num_heads)
  attn = dropout.dropout_layer(attn, drop, jax.random.PRNGKey(0))
  seq2 = seq + attn
  seq2 = jax.nn.standardize(seq2)

  #forward mlp
  logits = mlp.forward_mlp(mlp_params, seq2)
  
  #calc loss
  loss = softmax_entropy.softmax_cross_entropy(logits, target)
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
  mlp_layer = mlp.init_mlp(prng_key, d_model, d_ff)

  #create input and target
  initializer = jax.nn.initializers.normal(0.01)
  input = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)
  target = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

  #calc grad
  gradient = calc_grad(mlp_layer, input, target)
  print(gradient)

def main():
  test()
'''
if name == "main":
  main()'''