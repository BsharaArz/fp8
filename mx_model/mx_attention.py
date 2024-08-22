import jax
import jax.numpy as jnp
import mx
import train
from typing_extensions import NamedTuple

class Attention(NamedTuple):
  q: jax.Array
  k: jax.Array
  v: jax.Array

def init_attention(prng_key: jax.Array, batch_size: int, sequence_length: int, d_model: int):
  #initialize qkv
  cpu_0 = jax.devices('cpu')[0]
  with jax.default_device(cpu_0):
    initializer = jax.nn.initializers.normal(0.01)
    q = initializer(prng_key, (d_model, d_model), jnp.float32)
    k = initializer(prng_key, (d_model, d_model), jnp.float32)
    v = initializer(prng_key, (d_model, d_model), jnp.float32)

  return Attention(q, k, v)
  
def forward_attention(params: Attention, seq: jax.Array, num_heads):
  '''
  params: qkv in fp32
  seq: input seq in fp8 mx format
  '''
  #conduct multi head attention
  #multiply seq + qkv in fp8
  q = train.custom_matmul(seq, params.q)
  k = train.custom_matmul(seq, params.k)
  v = train.custom_matmul(seq, params.v)
  return multi_head_attention(q, k, v, num_heads) 

def scaled_dot_product(q: jax.Array, k: jax.Array, v: jax.Array):
  '''
  Attention Calculation
  attention = softmax[ (QK^T)/sqrt(dk) ] * V
  '''
  dk = k.shape[-1] #dim of k
  k = jnp.swapaxes(k, -2, -1)
  logits = train.custom_matmul(q, k) / jnp.sqrt(dk)
  weights = jax.nn.softmax(logits)
  output = train.custom_matmul(weights, v)
  return output

def multi_head_attention(q: jax.Array, k: jax.Array, v: jax.Array, num_heads: int):
  '''
  q, k, v given with shape (batch_size, seq_length, d_model)
  '''
  batch_size, seq_length, d_model = q.shape

  #reshape qkv based on num_heads -> (batch_size, seq_length, num_heads, d_head) -> transpose
  q = q.reshape(batch_size, seq_length, num_heads, d_model//num_heads).transpose(0, 2, 1, 3)
  k = k.reshape(batch_size, seq_length, num_heads, d_model//num_heads).transpose(0, 2, 1, 3)
  v = v.reshape(batch_size, seq_length, num_heads, d_model//num_heads).transpose(0, 2, 1, 3)

  #calc dot product of qkv and reshape
  values = scaled_dot_product(q, k, v)
  values = values.transpose(0, 2, 1, 3)
  values = values.reshape(batch_size, seq_length, d_model) #return to norm

  return values 