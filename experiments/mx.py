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
    '''s
    Params: seq in fp32 and dtype (either jnp.float8_e4m3fn or jnp.float8_e5m2)
    '''
    #calc scalar and scaled seq
    x = calc_scalar(seq, dt)
    if x == 0: #seq uses less exp than fp8 (no scale needed)
        return MX(seq.astype(dt), x)
    seq = seq/x
    
    #clip seq
    seq = jnp.clip(seq, min=(jnp.finfo(dt).min).astype(jnp.float32), max=(jnp.finfo(dt).max).astype(jnp.float32))
    
    #rewrite in dtype
    seq = seq.astype(dt)
    
    return MX(seq, x)

def mx_matmul(mx1: MX, mx2: MX):
    result = jnp.matmul(mx1.seq, mx2.seq)
    result = result * mx1.scalar if mx1.scalar != 0 else result
    result = result * mx2.scalar if mx2.scalar != 0 else result
    return result.astype(jnp.float32)


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