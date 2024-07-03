import jax
import jax.numpy as jnp

import attention
import mx

def forward_attention(params: attention.Attention, seq: jax.Array, num_heads):
  '''
  params: qkv in fp32
  seq: input seq in fp8 mx format
  '''
  #quantize seq
  seq = mx.quantize(seq, jnp.float8_e4m3fn)
  #conduct multi head attention
  #convert qkv to mx -> multiply in fp8 -> save as mx
  q = mx.mx_multiply(seq, mx.quantize(params.q, jnp.float8_e4m3fn))
  k = mx.mx_multiply(seq, mx.quantize(params.k, jnp.float8_e4m3fn))
  v = mx.mx_multiply(seq, mx.quantize(params.v, jnp.float8_e4m3fn))
  q = mx.quantize(q, jnp.float8_e4m3fn)
  k = mx.quantize(k, jnp.float8_e4m3fn)
  v = mx.quantize(v, jnp.float8_e4m3fn)
  return multi_head_attention(q, k, v, num_heads) #returns as fp8 mx format

def scaled_dot_product(q: mx.MX, k: mx.MX, v: mx.MX):
  '''
  Attention Calculation
  attention = softmax[ (QK^T)/sqrt(dk) ] * V
  '''
  dk = k.seq.shape[-1] #dim of k
  k = mx.mx_update(k, k.seq.swapaxes(-2, -1))
  logits = mx.mx_matmul(q, k) / jnp.sqrt(dk)
  weights = jax.nn.softmax(logits)
  weights = mx.quantize(weights, jnp.float8_e4m3fn)
  output = mx.mx_matmul(weights, v)
  return output

def multi_head_attention(q: mx.MX, k: mx.MX, v: mx.MX, num_heads: int):
  '''
  q, k, v given with shape (batch_size, seq_length, d_model)
  '''
  batch_size, seq_length, d_model = q.seq.shape

  #reshape qkv based on num_heads -> (batch_size, seq_length, num_heads, d_head) -> transpose
  q = mx.mx_update(q, q.seq.reshape(batch_size, seq_length, 8, d_model//8).transpose(0, 2, 1, 3))
  k = mx.mx_update(k, k.seq.reshape(batch_size, seq_length, 8, d_model//8).transpose(0, 2, 1, 3))
  v = mx.mx_update(v, v.seq.reshape(batch_size, seq_length, 8, d_model//8).transpose(0, 2, 1, 3))

  #calc dot product of qkv and reshape
  values = scaled_dot_product(q, k, v)
  values = values.transpose(0, 2, 1, 3)
  values = values.reshape(batch_size, seq_length, d_model) #return to norm

  return values 