import jax
import mx_transformer_block
from typing_extensions import NamedTuple

class Transformer(NamedTuple):
    blocks: list

def init_transformer(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks: int):
    #create a list of blocks of length num_blocks
    keys = jax.random.split(prng_key, num_blocks)
    blocks = [mx_transformer_block.init_block(keys[i], batch_size, sequence_length, d_model, d_ff) for i in range(num_blocks)]
    return Transformer(blocks)

def transformer_forward(model: Transformer, seq: jax.Array, num_heads, drop, prng_key):
    '''
    attempted to use jax.scan but failed due to mlp/attn layers being diff sizes
    got ValueError: scan got values with different leading axis sizes
    '''
    #instead using simple loop
    for block in model.blocks:
        seq = mx_transformer_block.block_forward(block, seq, num_heads, drop, prng_key)
    return seq