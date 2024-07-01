import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple

class MX(NamedTuple):
    seq: jax.Array
    scalar: int

def calc_scalar(seq: jax.Array, dt):
    '''
    Return an int8 that represents the scalar of the mx format of seq
    '''
    #algorithm 1 from paper
    e_max_elem = jnp.finfo(dt).maxexp
    shared_exp = jnp.floor(jnp.log2(jnp.max(jnp.abs(seq)))) - e_max_elem
    x = 2.0 ** shared_exp
    
    #clip x
    x = jnp.clip(x, min=jnp.iinfo(jnp.int8).min, max=jnp.iinfo(jnp.int8).max)
    return x.astype(jnp.int8)
 
def quantize(seq: jax.Array, dt):
    '''
    Params: seq in fp32 and dtype (either jnp.float8_e4m3fn or jnp.float8_e5m2)
    '''
    #calc scalar and scaled seq
    x = calc_scalar(seq, dt)
    return jax.lax.cond(x == 0, no_scale, scale, *(seq, x))

#-------------------------------------------
# condition funcs for jax.lax.cond
def no_scale(seq: jax.Array, x):
    return MX(seq.astype(jnp.float8_e4m3fn), x)

def scale(seq: jax.Array, x):
    seq = seq/x
    
    #clip seq
    seq = jnp.clip(seq, min=(jnp.finfo(jnp.float8_e4m3fn).min).astype(jnp.float32), max=(jnp.finfo(jnp.float8_e4m3fn).max).astype(jnp.float32))
    
    #rewrite in dtype
    seq = seq.astype(jnp.float8_e4m3fn)
    
    return MX(seq, x)
#-------------------------------------------

def mx_matmul(mx1: MX, mx2: MX):
    result = jnp.matmul(mx1.seq, mx2.seq)
    result = jax.lax.cond(mx1.scalar == 0, no_change, apply_scale, *(result, mx1.scalar))
    result = jax.lax.cond(mx2.scalar == 0, no_change, apply_scale, *(result, mx2.scalar))
    return result.astype(jnp.float32)

def mx_multiply(mx1: MX, mx2: MX):
    result = jnp.multiply(mx1.seq, mx2.seq)
    result = jax.lax.cond(mx1.scalar == 0, no_change, apply_scale, *(result, mx1.scalar))
    result = jax.lax.cond(mx2.scalar == 0, no_change, apply_scale, *(result, mx2.scalar))
    return result.astype(jnp.float32)

#-------------------------------------------
# condition funcs for jax.lax.cond
def no_change(result, scalar):
    return result

def apply_scale(result, scalar):
    return result * scalar
#-------------------------------------------

def mx_update(mx1: MX, newSeq: jax.Array): #update seq w/o changing scalar (transpose, reshape, etc.)
    return MX(newSeq, mx1.scalar)

def test():
    #initialize params
    prng_key = jax.random.PRNGKey(0)
    d_model = 512
    d_ff = 1024
    batch_size = 32
    sequence_length = 16

    #initialize seq
    initializer = jax.nn.initializers.normal(0.01)
    seq = initializer(prng_key, (batch_size, sequence_length, d_model), jnp.float32)
    quantized_seq = quantize(seq, jnp.float8_e4m3fn)
    print(jnp.matmul(seq, seq.swapaxes(-2, -1))[0][0])
    transposed = MX(quantized_seq.seq.swapaxes(-2, -1), quantized_seq.scalar)
    print(mx_matmul(quantized_seq, transposed)[0][0])

def main():
    test() 

# if name == "main":
#     main()