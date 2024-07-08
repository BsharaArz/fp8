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
    x = jnp.clip(x, min=jnp.iinfo(jnp.int8).min, max=jnp.iinfo(jnp.int8).max) #TODO: test uint8 (since can never be negative)
    return x.astype(jnp.int8)

def quantize(seq: jax.Array, dt):
    '''
    Params: seq in fp32 and dtype (either jnp.float8_e4m3fn or jnp.float8_e5m2)
    '''
    #calc scalar and scaled seq
    x = calc_scalar(seq, dt)
    if dt == jnp.float8_e4m3fn:
        return jax.lax.cond(x == 0, no_scale, scale, *(seq, x))
    elif dt == jnp.float8_e5m2:
        return jax.lax.cond(x == 0, grad_no_scale, grad_scale, *(seq, x))

#---------------------------------
# condition funcs for jax.lax.cond
def no_scale(seq: jax.Array, x):
    return MX(jnp.clip(seq, min=(jnp.finfo(jnp.float8_e4m3fn).min).astype(jnp.float32), max=(jnp.finfo(jnp.float8_e4m3fn).max).astype(jnp.float32)).astype(jnp.float8_e4m3fn), jnp.int8(1))

def scale(seq: jax.Array, x):
    seq = seq/x
    
    #clip seq
    seq = jnp.clip(seq, min=(jnp.finfo(jnp.float8_e4m3fn).min).astype(jnp.float32), max=(jnp.finfo(jnp.float8_e4m3fn).max).astype(jnp.float32))
    
    #rewrite in dtype
    seq = seq.astype(jnp.float8_e4m3fn)
    
    return MX(seq, x)

def grad_no_scale(seq: jax.Array, x):
    return MX(seq.astype(jnp.float8_e4m3fn), jnp.int8(1))

def grad_scale(seq: jax.Array, x):
    seq = seq/x
    
    #clip seq
    seq = jnp.clip(seq, min=(jnp.finfo(jnp.float8_e5m2).min).astype(jnp.float32), max=(jnp.finfo(jnp.float8_e5m2).max).astype(jnp.float32))
    
    #rewrite in dtype
    seq = seq.astype(jnp.float8_e5m2)
    
    return MX(seq, x)
#---------------------------------

'''
--------------------
MULTIPLICATION FUNCS
'''

def mx_matmul(mx1: MX, mx2: MX): #converted to fp32 since cpu support fp8 mult
    result = (jnp.matmul(mx1.seq.astype(jnp.float32), mx2.seq.astype(jnp.float32), preferred_element_type=jnp.float32) * mx1.scalar) * mx2.scalar
    return result.astype(jnp.float32)

def mx_multiply(mx1: MX, mx2: MX):
    result = (jnp.multiply(mx1.seq.astype(jnp.float32), mx2.seq.astype(jnp.float32)) * mx1.scalar) * mx2.scalar
    return result.astype(jnp.float32)


#UPDATE SEQ
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
    seq = initializer(prng_key, (batch_size, sequence_length), jnp.float32)
    quantized_seq = quantize(seq, jnp.float8_e4m3fn)

    print(jnp.matmul(seq, seq.swapaxes(-2, -1))[0])
    transposed = MX(quantized_seq.seq.swapaxes(-2, -1), quantized_seq.scalar)
    print(mx_matmul(quantized_seq, transposed)[0])

def main():
    test() 

if __name__ == "__main__":
    main()