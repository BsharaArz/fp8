import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple

import attention
import mx_attention
import mlp
import mx_mlp
import mx
import transformer_block
import functools

#JUST MLP FOR NOW
class TransformerBlock(NamedTuple):
  attn_layer: attention.Attention
  mlp_layer: mlp.MLP

def init_block(prng_key, batch_size, sequence_length, d_model, d_ff):
  #initialize mlp
  key0, key1 = jax.random.split(prng_key)
  attn_layer = attention.init_attention(key0, batch_size, sequence_length, d_model)
  mlp_layer = mlp.init_mlp(key1, d_model, d_ff)

  return TransformerBlock(attn_layer, mlp_layer)

# LAYERNORM in fp32

def dropout(seq, drop, prng_key):
  '''
  dropout function given a seq and dropout rate
  '''
  #compute dropout
  mask = jax.random.uniform(prng_key, seq.shape) > drop
  #quantize arrays
  mask = mx.quantize(jnp.asarray(mask, jnp.float32), jnp.float8_e4m3fn)
  seq = mx.quantize(seq, jnp.float8_e4m3fn)
  return mx.mx_multiply(mask, seq) / (1.0 - drop)

# @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
def block_forward(params: TransformerBlock, seq:jax.Array, num_heads, drop, prng_key):
  '''
  conduct a forward pass for a singular transformer block
  '''
  #layer norm
  seq = transformer_block.normalize(seq)

  #forward attention
  attn = mx_attention.forward_attention(params.attn_layer, seq, num_heads) #return fp32

  #dropout
  attn = dropout(attn, drop, prng_key)

  #residual connection
  seq = seq + attn
  
  #layer norm
  seq = transformer_block.normalize(seq)

  #forward mlp
  logits = mx_mlp.forward_mlp(params.mlp_layer, seq) #return fp32

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

  print("FP32:")
  l = mlp.forward_mlp(block.mlp_layer, seq)
  print(seq + l)

  print("FP8:")
  print(output)

def main():
  test()


# if name == "main":
#   main()