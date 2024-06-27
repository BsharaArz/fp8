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
    shared_exp = (jnp.log2(jnp.max(jnp.abs(seq)))).astype(int) - e_max_elem
    x = 2 ** jnp.abs(shared_exp)
    
    #clip x
    x = jnp.clip(x, min=jnp.iinfo(jnp.int8).min, max=jnp.iinfo(jnp.int8).max)
    return x
    
def quantize(seq: jax.Array, dt):
    '''
    Params: seq in fp32 and dtype (either jnp.float8_e4m3fn or jnp.float8_e5m2)
    '''
    #calc scalar and scaled seq
    x = calc_scalar(seq, dt)
    seq = seq*x
    
    #clip seq
    seq = jnp.clip(seq, min=(jnp.finfo(dt).min).astype(jnp.float32), max=(jnp.finfo(dt).max).astype(jnp.float32))
    
    #rewrite in dtype
    seq = seq.astype(dt)
    
    return MX(seq, x)

def mx_matmul(mx1: MX, mx2: MX):
    return jnp.matmul(mx1.seq, mx2.seq, preferred_element_type=jnp.float32)/(float(mx1.scalar) * float(mx2.scalar))

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
if name == "main":
    main()