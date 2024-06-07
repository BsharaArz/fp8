import jax 
import jax.numpy as jnp
import mlp
import softmax_entropy

def calc_loss(params: mlp.MLP, input: jax.Array, target: jax.Array):
  #pass through forward, calculate loss compared w/ target
  logits = mlp.forward_mlp(params, input)
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

if name == "main":
  main()