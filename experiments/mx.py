import jax
import jax.numpy as jnp
import numpy as np
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
@jax.custom_gradient
def tensor_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):
    def grad(g):
        _, matmul_grad = jax.vjp(jnp.matmul, tensor1, tensor2)
        return matmul_grad(g)
    mx1 = quantize(tensor1)
    mx2 = quantize(tensor2)
    return mx_matmul(mx1, mx2), grad

@jax.custom_gradient
def tensor_multiply(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):
    def grad(g):
        _, mult_grad = jax.vjp(jnp.multiply, tensor1, tensor2)
        return mult_grad(g)
    mx1 = quantize(tensor1)
    mx2 = quantize(tensor2)
    return mx_multiply(mx1, mx2), grad

#Block level quantization + matmul
def block_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):  
    k = min(tensor1.shape[-1], k)
    mx1_k = quantize_k_left(tensor1, k)
    mx2_k = quantize_k_right(tensor2, k)  
    return mx_matmul_k(mx1_k, mx2_k, k)

def block_multiply(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):  
    k = min(tensor1.shape[-1], k)
    mx1_k = quantize_k_left(tensor1, k)
    mx2_k = quantize_k_right(tensor2, k)  
    return mx_multiply_k(mx1_k, mx2_k)
    
@jax.custom_gradient
def kernel_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int = 128):
    def grad(g):
        _, matmul_grad = jax.vjp(jnp.matmul, tensor1, tensor2)
        return matmul_grad(g)
    return kernels.kernel_blocked_matmul(tensor1, tensor2, 128), grad


'''
--------------------
SCALE ALG FROM PAPER
'''
def calc_scalar(tensor: jax.Array):
    '''
    Return an int8 that represents the scalar of the mx format of seq
    '''
    #algorithm 1 from paper
    shared_exp = jnp.floor(jnp.log2(jnp.max(jnp.abs(tensor)))) - 9
    x = 2.0 ** shared_exp

    #clip x 
    # x = jnp.clip(x, a_min=1, a_max=127)
    #quicker to do this than jnp.clip 
    x = jnp.minimum(jnp.maximum(x, 1), 127)
    return x.astype(jnp.int8)

'''
--------------------
TENSOR LEVEL HELPERS
'''

def scale(tensor: jax.Array, x):
    tensor = tensor/x
    
    #clip tensor - REMOVE or look for alternative
    # tensor = jnp.clip(tensor, a_min=(jnp.finfo(jnp.float8_e4m3fn).min).astype(jnp.float32), a_max=(jnp.finfo(jnp.float8_e4m3fn).max).astype(jnp.float32))
    
    #rewrite in dtype
    # tensor = tensor.astype(jnp.bfloat16)
    
    return MX(tensor, x)

def quantize(tensor: jax.Array):
    '''
    Params: tensor in fp32 -> to be converted to mx format
    '''
    #calc scalar and scaled tensor
    scalar = calc_scalar(tensor)
    return scale(tensor, scalar)
    # return MX(tensor/scalar, scalar)

def mx_matmul(mx1: MX, mx2: MX): #converted to fp32 since cpu support fp8 mult
    result = jnp.matmul(mx1.seq.astype(jnp.bfloat16), mx2.seq.astype(jnp.bfloat16)) * (mx1.scalar * mx2.scalar)
    return result.astype(jnp.float32)

def mx_multiply(mx1: MX, mx2: MX):
    result = jnp.multiply(mx1.seq, mx2.seq) * (mx1.scalar * mx2.scalar)
    return result.astype(jnp.float32)

#UPDATE SEQ
def mx_update(mx1: MX, newSeq: jax.Array): #update seq w/o changing scalar (transpose, reshape, etc.)
    return MX(newSeq, mx1.scalar)

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
    # print(tensor_scales.shape)
    return tensor_scales

def calc_scalar_k_right(tensor: jax.Array, k: int):
    # print(tensor.shape)
    # print(tensor.shape[:-2])

    new_shape = [*tensor.shape[:-2], tensor.shape[-2]//k, k, tensor.shape[-1]]
    # print(new_shape)
    reshaped = tensor.reshape(*new_shape)
    tensor_scales = jnp.apply_along_axis(calc_scalar, -2, reshaped)
    # print(tensor_scales.shape)
    return tensor_scales

def scale_k_left(tensor: jax.Array, scales: jax.Array, k: int):
    # cpu_0 = jax.devices('cpu')[0]
    with jax.default_device(cpu_0):
        scales_expanded = jnp.repeat(scales, k, axis=-1)
    return tensor/scales_expanded

def scale_k_right(tensor: jax.Array, scales: jax.Array, k: int):
    # cpu_0 = jax.devices('cpu')[0]
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
def twod(x: jax.Array, y: jax.Array, scalex: jax.Array, scaley: jax.Array, bk: int = 512):
    m, k = x.shape
    _, n = y.shape

    z = jnp.zeros((m, n))
    for k_i in range(scaley.shape[0]):
        x_block = jax.lax.dynamic_slice(x, [0, k_i * bk], [m, bk])
        y_block = jax.lax.dynamic_slice(y, [k_i * bk, 0], [bk, n])
        x_scale = jax.lax.dynamic_slice(scalex, [0, k_i], [m, 1])
        y_scale = jax.lax.dynamic_slice(scaley, [k_i , 0], [1, n])

        new = jnp.matmul(x_block.astype(jnp.bfloat16), y_block.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16) 
        x_scale = jnp.broadcast_to(x_scale, new.shape)
        y_scale = jnp.broadcast_to(y_scale, new.shape)
        new = jnp.multiply(new, jnp.multiply(x_scale, y_scale))# * jax.numpy.broadcast_to(y_scale, new.shape) #* jnp.expand_dims(scalex[:, k_i]), -1) * scaley[k_i, :]
        z = jnp.add(z, new)
    # for m_i in range(m):
    #     for n_i in range(n):
    #         new = 0
    #         for k_i in range(scaley.shape[0]):
    #             cpu_0 = jax.devices('cpu')[0]
    #             with jax.default_device(cpu_0):
    #                 x_block = jax.lax.dynamic_slice(x, [m_i, k_i * bk], [1, bk])
    #                 y_block = jax.lax.dynamic_slice(y, [k_i * bk, n_i], [bk, 1])
    #             new += jnp.matmul(x_block.astype(jnp.float32), y_block.astype(jnp.float32), preferred_element_type=jnp.float32) * scalex[m_i, k_i] * scaley[k_i, n_i]
    #         z = z.at[(m_i, n_i)].set(new.astype(jnp.float32)[0][0])
    return z

@functools.partial(jax.jit, static_argnames=["k"])
def blocked_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int, scale1: jax.Array, scale2: jax.Array):
    '''
    Blocked matrix multiplication for n-d arrays
    - assume tensor1 + scale1 n-dim, tensor2 + scale2 2-dim
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
        # print("we are 2d")
        # result = np.zeros([tensor1.shape[0],tensor2.shape[1]], dtype='bfloat16')
        # return jnp.array(kernels.nki_matmul_fully_optimized_(np.array(tensor1.T), np.array(tensor2), result))
        return twod(tensor1, tensor2, scale1, scale2, k)
    return result

def mx_matmul_k(mx1: MX_K, mx2: MX_K, k: int):
    return jnp.array(blocked_matmul(mx1.seq.astype(jnp.float32), mx2.seq.astype(jnp.float32), k, mx1.scalar, mx2.scalar))


# @functools.partial(jax.jit, static_argnames=["k"])
# def one_element(row: jax.Array, col: jax.Array, k: int, scale1: jax.Array, scale2: jax.Array):
#     r = row.reshape(-1, k)
#     c = col.reshape(-1, k)
#     result = jnp.einsum('ik,ik->i', r, c) * scale1 * scale2
#     return result

# @functools.partial(jax.jit, static_argnames=["k"])
# def twod(tensor1: jax.Array, tensor2: jax.Array, k: int, scale1: jax.Array, scale2: jax.Array):
#     m, n = tensor1.shape[0], tensor2.shape[1]
    
#     def body_fun(i, val):
#         row = tensor1[i]
#         result_row = jax.vmap(lambda j: one_element(row, tensor2[:, j], k, scale1[i], scale2[:, j]))(jnp.arange(n))
#         return val.at[i].set(result_row)
    
#     result = jax.lax.fori_loop(0, m, body_fun, jnp.zeros((m, n)))
#     return result

# @functools.partial(jax.jit, static_argnames=["k"])
# def blocked_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int, scale1: jax.Array, scale2: jax.Array):
#     if tensor1.ndim > 2:
#         return jax.vmap(lambda t1, s1: blocked_matmul(t1, tensor2, k, s1, scale2))(tensor1, scale1)
#     else:
#         return twod(tensor1, tensor2, k, scale1, scale2)

# @jax.jit
# def mx_matmul_k(mx1: MX_K, mx2: MX_K):
#     k = mx2.seq.shape[0] // mx2.scalar.shape[0]
#     return blocked_matmul(mx1.seq.astype(jnp.bfloat16), mx2.seq.astype(jnp.bfloat16), k, mx1.scalar, mx2.scalar)

def mx_multiply_k(mx1: MX_K, mx2: MX_K):
    k = mx1.seq.shape[-1]//mx2.scalar.shape[-1]
    return jnp.multiply(mx1.seq, mx2.seq) * jnp.repeat(mx1.scalar, k, axis=-1) * jnp.repeat(mx2.scalar, k, axis=0)

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
    with jax.named_scope("REGMATMUL"):
        result = jnp.matmul(seq1, seq2)
    return result


if __name__ == "__main__":
    with jax.default_device(cpu_0):
        initializer = jax.nn.initializers.normal(0.01)
        seq1 = initializer(jax.random.PRNGKey(0), (256, 512), jnp.float32)
        seq2 = initializer(jax.random.PRNGKey(0), (512, 1024), jnp.float32)
    result = regmatmul(seq1, seq2)
    print(result)
