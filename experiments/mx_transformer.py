import jax
import transformer
import mlp
import mx_transformer_block

#init transformer with MX blocks, forward + abstraction remains the same

def init_transformer(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks: int):
    #create a list of blocks of length num_blocks
    keys = jax.random.split(prng_key, num_blocks)
    blocks = [mx_transformer_block.init_block(keys[i], batch_size, sequence_length, d_model, d_ff) for i in range(num_blocks)]
    return transformer.Transformer(blocks)

def transformer_forward(model: transformer.Transformer, seq: jax.Array, num_heads, drop, prng_key):
    '''
    attempted to use jax.scan but failed due to mlp/attn layers being diff sizes
    got ValueError: scan got values with different leading axis sizes
    '''
    #instead using simple loop
    for block in model.blocks:
        if num_heads == 0: #indicator of fp32 - doing just for testing
            l = mlp.forward_mlp(block.mlp_layer, seq)
            seq = seq+l
        else:
            seq = mx_transformer_block.block_forward(block, seq, num_heads, drop, prng_key)
    return seq