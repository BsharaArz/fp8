import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple

import attention
import mlp

class TransformerBlock(NamedTuple):
  attn_layer: attention.Attention
  mlp_layer: mlp.MLP

def init_block(prng_key, batch_size, sequence_length, d_model, d_ff):
  #initialize mlp/attention layers
  key0, key1 = jax.random.split(prng_key)
  attn_layer = attention.init_attention(key0, batch_size, sequence_length, d_model)
  mlp_layer = mlp.init_mlp(key1, d_model, d_ff)

  return TransformerBlock(attn_layer, mlp_layer)

def normalize(seq: jax.Array, epsilon=1e-6):
  '''
  layer normalization of a given seq using formula
  '''
  #compute mean/var
  mean = seq.mean(-1, keepdims=True)
  var = seq.var(-1, keepdims=True)

  #compute norm
  normalized = (seq - mean) / jnp.sqrt(var + epsilon)

  #compute scale/shift
  scale = jnp.ones_like(mean)
  shift = jnp.zeros_like(mean)

  #formula
  output = normalized * scale + shift
  return output

def dropout(seq, drop, prng_key):
  '''
  dropout function given a seq and dropout rate
  '''
  #compute dropout
  assert 0 <= drop <= 1
  if drop == 1: return jnp.zeros_like(seq)
  mask = jax.random.uniform(prng_key, seq.shape) > drop
  return jnp.asarray(mask, jnp.float32) * seq / (1.0 - drop)

def block_forward(params: TransformerBlock, seq:jax.Array, num_heads, drop, prng_key):
  '''
  conduct a forward pass for a singular transformer block
  '''
  #layer norm
  seq = normalize(seq)

  #forward attention
  attn, _ = attention.forward_attention(params.attn_layer, num_heads)

  #dropout
  attn = dropout(attn, drop, prng_key)

  #residual connection
  seq = seq + attn

  #layer norm
  seq = normalize(seq)

  #forward mlp
  logits = mlp.forward_mlp(params.mlp_layer, seq)

  #dropout
  logits = dropout(logits, drop, prng_key)

  #residual connection
  seq = seq + logits

  return seq

def test():
  #initialize params
  prng_key = jax.random.PRNGKey(0)
  d_model = 512
  d_ff = 1024
  batch_size = 32
  sequence_length = 16
  num_heads = 8
  drop = 0.5

  #initialize block
  block = init_block(prng_key, batch_size, sequence_length, d_model, d_ff)

  #initialize seq
  initializer = jax.nn.initializers.normal(0.01)
  seq = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

  output = block_forward(block, seq, num_heads, drop, prng_key)

  print(seq)
  print(output)

def main():
  test()

if name == "main":
  main()