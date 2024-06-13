import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple
import collections

import transformer_block
import attention
import mlp

class Transformer(NamedTuple):
    blocks: list

def init_transformer(block: transformer_block.TransformerBlock, num_blocks: int):
    #create a list of blocks of length num_blocks
    return Transformer([block for _ in range(num_blocks)])

def transformer_forward(model: Transformer, seq: jax.Array, num_heads, drop, prng_key):
    '''
    attempted to use jax.scan but failed due to mlp/attn layers being diff sizes
    got ValueError: scan got values with different leading axis sizes
    '''
    #instead using simple loop
    for block in model.blocks:
        seq = transformer_block.block_forward(block, seq, num_heads, drop, prng_key)
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
    num_blocks = 12

    #initialize mlp/attention layers
    attn_layer = attention.init_attention(prng_key, batch_size, sequence_length, d_model)
    mlp_layer = mlp.init_mlp(prng_key, d_model, d_ff)

    #initialize block
    block = transformer_block.init_block(attn_layer, mlp_layer)

    #initialize transformer
    transform = init_transformer(block, num_blocks)

    #initialize seq
    initializer = jax.nn.initializers.normal(0.01)
    seq = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)

    output = transformer_forward(transform, seq, num_heads, drop, prng_key)
    
    print(seq)
    print(output)

def main():
    test()

if name == "main":
    main()
