import jax
import transformer
import mlp
import mx_transformer_block
import transformer_block
import attention

#init transformer with MX blocks, abstraction remains the same

def transformer_forward(model: transformer.Transformer, seq: jax.Array, num_heads, drop, prng_key):
    '''
    attempted to use jax.scan but failed due to mlp/attn layers being diff sizes
    got ValueError: scan got values with different leading axis sizes
    '''
    #instead using simple loop
    for block in model.blocks:
        seq = mx_transformer_block.block_forward(block, seq, num_heads, drop, prng_key)
    return seq