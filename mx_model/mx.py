import jax
import jax.numpy as jnp
from typing_extensions import NamedTuple
import functools
import kernels

cpu_0 = jax.devices('cpu')[0]

class MX(NamedTuple):
    seq: jax.Array
    scalar: int

class MX_K(NamedTuple):
    seq: jax.Array
    scalar: jax.Array

'''
--------------------
MAIN FUNCS
'''
#Tensor level quantization + matmul
def tensor_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):
    mx1 = quantize(tensor1)
    mx2 = quantize(tensor2)
    return mx_matmul(mx1, mx2)

#Block level quantization + matmul
@jax.custom_gradient
def block_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):  
    def grad(g):
        _, matmul_grad = jax.vjp(jnp.matmul, tensor1, tensor2)
        return matmul_grad(g)
    k = min(tensor1.shape[-1], k)
    mx1_k = quantize_k_left(tensor1, k)
    mx2_k = quantize_k_right(tensor2, k)  
    return mx_matmul_k(mx1_k, mx2_k, k), grad

#Block level quantization + matmul fused in kernel
def kernel_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):
    return kernels.kernel_blocked_matmul(tensor1, tensor2, 128)


'''
--------------------
SCALE ALG FROM PAPER
'''
def calc_scalar(tensor: jax.Array):
    '''
    Return an int8 that represents the scalar of the mx format of tensor
    '''
    #algorithm 1 from paper
    shared_exp = jnp.floor(jnp.log2(jnp.max(jnp.abs(tensor)))) - 7
    x = 2.0 ** shared_exp

    #clip to pos int8 range 
    x = jnp.minimum(jnp.maximum(x, 1), 127)
    return x.astype(jnp.int8)

'''
--------------------
TENSOR LEVEL HELPERS
'''

def scale(tensor: jax.Array, x):
    tensor = tensor/x
    
    #clip seq
    tensor = jnp.clip(tensor, min=-240, max=240)
    
    #rewrite in dtype
    tensor = tensor.astype(jnp.bfloat16)
    
    return MX(tensor, x)

def quantize(tensor: jax.Array):
    '''
    Params: tensor in fp32 -> to be converted to mx format
    '''
    #calc scalar and scaled tensor
    scalar = calc_scalar(tensor)
    return scale(tensor, scalar)

@jax.custom_gradient
def mx_matmul(mx1: MX, mx2: MX): 
    '''
    Forward: Conduct matmul on the quantized tensors, then apply the scales
    Backward: Calculate the gradient of the matmul on, then apply the scales
    '''
    def grad(g):
        _, matmul_grad = jax.vjp(jnp.matmul, mx1.seq.astype(jnp.bfloat16) * mx1.scalar, mx2.seq.astype(jnp.bfloat16) * mx2.scalar) 
        return matmul_grad(g)
    result = jnp.matmul(mx1.seq.astype(jnp.bfloat16), mx2.seq.astype(jnp.bfloat16)) * mx1.scalar * mx2.scalar
    return result, grad

'''
--------------------
BLOCK LEVEL HELPERS
'''
def calc_scalar_k_left(tensor: jax.Array, k: int):
    if k == tensor.shape[:-1]:
        reshaped = tensor.reshape(*tensor.shape[:-1], 1, k)
    else:
        reshaped = tensor.reshape(*tensor.shape[:-1], -1, k)
    tensor_scales = jnp.apply_along_axis(calc_scalar, -1, reshaped)
    return tensor_scales

def calc_scalar_k_right(tensor: jax.Array, k: int):
    new_shape = [*tensor.shape[:-2], tensor.shape[-2]//k, k, tensor.shape[-1]]
    reshaped = tensor.reshape(*new_shape)
    tensor_scales = jnp.apply_along_axis(calc_scalar, -2, reshaped)
    return tensor_scales

def scale_k_left(tensor: jax.Array, scales: jax.Array, k: int):
    with jax.default_device(cpu_0):
        scales_expanded = jnp.repeat(scales, k, axis=-1)
    return tensor/scales_expanded

def scale_k_right(tensor: jax.Array, scales: jax.Array, k: int):
    with jax.default_device(cpu_0):
        scales_expanded = jnp.repeat(scales, k, axis=-2)
    return tensor/scales_expanded

def quantize_k_left(tensor: jax.Array, k: int = 32):
    scales = calc_scalar_k_left(tensor, k)
    scaled_tensor = scale_k_left(tensor, scales, k)
    return MX_K(scaled_tensor, scales)

def quantize_k_right(tensor: jax.Array, k: int = 32):
    scales = calc_scalar_k_right(tensor, k)
    scaled_tensor = scale_k_right(tensor, scales, k)
    return MX_K(scaled_tensor, scales)

@functools.partial(jax.jit, static_argnames=['bk'])
def twod(x: jax.Array, y: jax.Array, scalex: jax.Array, scaley: jax.Array, bk: int = 128):
    '''
    Blocked matrix multiplication for 2-d arrays
    '''
    m, k = x.shape
    _, n = y.shape
    bk = min(k, 128)

    z = jnp.zeros((m, n))
    for k_i in range(k//bk):
        x_block = jax.lax.dynamic_slice(x, [0, k_i * bk], [m, bk])
        y_block = jax.lax.dynamic_slice(y, [k_i * bk, 0], [bk, n])
        x_scale = jax.lax.dynamic_slice(scalex, [0, k_i], [m, 1])
        y_scale = jax.lax.dynamic_slice(scaley, [k_i , 0], [1, n])

        new = jnp.matmul(x_block.astype(jnp.bfloat16), y_block.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16) 
        with jax.default_device(cpu_0):
            x_scale = jnp.repeat(x_scale, n, axis=-1)
            y_scale = jnp.repeat(y_scale, m, axis=0)
        new = new * x_scale * y_scale 
        z = jnp.add(z, new)
    return z

@functools.partial(jax.jit, static_argnames=["k"])
def blocked_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int, scale1: jax.Array, scale2: jax.Array):
    '''
    Blocked matrix multiplication for n-d arrays
    '''
    result_shape = [*tensor1.shape[:-1], tensor2.shape[-1]]
    result = jnp.zeros(result_shape)
    if(len(tensor2.shape) > 2):
        for x in range(tensor2.shape[0]):
            new = blocked_matmul(tensor1[x], tensor2[x], k, scale1[x], scale2[x])
            result = result.at[x].set(new)
        return result 
    elif(len(tensor1.shape) > 2):
        for x in range(tensor1.shape[0]):
            result = result.at[x].set(blocked_matmul(tensor1[x], tensor2, k, scale1[x], scale2))
    if(len(tensor1.shape) <= 2 and len(tensor2.shape) <= 2):
        return twod(tensor1, tensor2, scale1, scale2, k)
    return result

def mx_matmul_k(mx1: MX_K, mx2: MX_K, k: int):
    return jnp.array(blocked_matmul(mx1.seq.astype(jnp.float32), mx2.seq.astype(jnp.float32), k, mx1.scalar, mx2.scalar))


#----------------------TESTS------------------------

def test():
    #initialize params
    prng_key = jax.random.PRNGKey(0)
    d_model = 512
    d_ff = 1024
    batch_size = 32
    sequence_length = 16

    #initialize seq
    cpu_0 = jax.devices('cpu')[0]
    with jax.default_device(cpu_0):
        initializer = jax.nn.initializers.normal(0.01)
        seq1 = initializer(prng_key, (2, 4, 8), jnp.float32)
        seq2 = initializer(prng_key, (2, 8, 4), jnp.float32)

    mult32 = jnp.matmul(seq1, seq2)
    print(mult32.shape)
    
    # seq1_nok = quantize(seq1)
    # seq2_nok = quantize(seq2)
    # mult_nok = mx_matmul(seq1_nok, seq2_nok)
    # seq2 = jax.lax.broadcast_in_dim(seq2, [*seq1.shape[:-2], *seq2.shape[-2:]])
    mult8_k = block_matmul(seq1, seq2, 2)
    print(mult8_k.shape)
    print(mult32)
    # print(mult_nok[0][0][:10])
    print(mult8_k)
    # print(jnp.matmul(seq1[0], seq2[0]))

def main():
    test() 

def test2():
    seq1 = jnp.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    scale1 = jnp.array([[[2], [3]], [[2], [3]]])
    seq2 = jnp.array([[3, 4], [3, 4]])
    scale2 = jnp.array([[1, 2]])

    print(blocked_matmul(seq1, seq2, 2, scale1, scale2))

def test3():
    seq1 = jnp.array([[1, 2], [1, 2]])
    print(len(seq1.shape))

@jax.jit
def namedscope(seq1, seq2):
    with jax.named_scope("NO KERNEL"):
        mx1_k = quantize_k_left(seq1, 128)
        mx2_k = quantize_k_right(seq2, 128)  
        result=mx_matmul_k(mx1_k, mx2_k, 128)
    return result

def profiletest():
    with jax.default_device(cpu_0):
        initializer = jax.nn.initializers.normal(0.01)
        seq1 = initializer(jax.random.PRNGKey(0), (256, 512), jnp.float32)
        seq2 = initializer(jax.random.PRNGKey(0), (512, 1024), jnp.float32)
    
    result = namedscope(seq1, seq2)
    print(result)
    
@jax.jit
def regmatmul(seq1, seq2):
    return seq1


if __name__ == "__main__":
    with jax.default_device(cpu_0):
        initializer = jax.nn.initializers.normal(0.01)
        seq1 = initializer(jax.random.PRNGKey(0), (256, 512), jnp.float32)
        seq2 = initializer(jax.random.PRNGKey(0), (512, 1024), jnp.float32)
    result = regmatmul(seq1, seq2)
    print(result)
