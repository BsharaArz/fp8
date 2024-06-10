import jax
import jax.numpy as jnp

def scaled_dot_product(q, k, v):
  '''
  Attention Calculation
  attention = softmax[ (QK^T)/sqrt(dk) ] * V
  '''
  dk = k.shape[-1] #dim of k
  logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(dk)
  weights = jax.nn.softmax(logits)
  output = jnp.matmul(weights, v)
  return output, weights

def multi_head_attention(q, k, v, num_heads):
  '''
  q, k, v given with shape (batch_size, seq_length, d_model)
  '''
  batch_size, seq_length, d_model = q.shape

  #reshape qkv based on num_heaeds
  q = q.reshape(batch_size, seq_length, num_heads, -1) #(batch_size, seq_length, num_heads, d_head)
  k = k.reshape(batch_size, seq_length, num_heads, -1)
  v = v.reshape(batch_size, seq_length, num_heads, -1)

  q = q.transpose(0, 2, 1, 3)
  k = k.transpose(0, 2, 1, 3)
  v = v.transpose(0, 2, 1, 3)

  #calc dot product of qkv and reshape
  values, attention = scaled_dot_product(q, k, v)
  values = values.transpose(0, 2, 1, 3)
  values = values.reshape(batch_size, seq_length, d_model) # return to norm

  return values, attention

def test():
  #initialize params
  prng_key = jax.random.PRNGKey(0)
  d_model = 512
  batch_size = 32
  sequence_length = 16

  #initialize qkv
  initializer = jax.nn.initializers.normal(0.01)
  q = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)
  k = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)
  v = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

  #pass through attention
  num_heads = 8
  values, attention = multi_head_attention(q, k, v, num_heads)

  #check shape
  print(values.shape)
  print(attention.shape)

def main():
  test()

if name == "main":
  main()