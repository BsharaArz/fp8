'''import mx
import jax
from jax import numpy as jnp

@jax.jit
def foo():
    prng_key = jax.random.PRNGKey(0)
    cpu_0 = jax.devices('cpu')[0]
    with jax.default_device(cpu_0):
        initializer = jax.nn.initializers.normal(0.01)
        seq = initializer(prng_key, (1, 512, 256), jnp.float32)
        seq2 = initializer(prng_key, (256, 512), jnp.float32)

    with jax.named_scope("QUANTIZATION"):
        seq = mx.quantize(seq)
        seq2 = mx.quantize(seq2)
    with jax.named_scope("MATMUL"):
        mult = mx.mx_matmul(seq, seq2)
    return mult

for i in range(1):
    x = foo()
'''
'''
prng_key = jax.random.PRNGKey(0)
initializer = jax.nn.initializers.normal(0.01)
seq = initializer(prng_key, (1, 512, 256), jnp.float32)
for x in seq:
    for y in x:
        print(y.shape)'''