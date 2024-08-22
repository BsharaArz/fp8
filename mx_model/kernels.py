import jax
from jax import numpy as jnp
import numpy as np
import neuronxcc
import scipy
import neuronxcc.nki
import neuronxcc.nki.language as nl
from neuronxcc.nki import tensor
from neuronxcc.nki.distributed.collectives import collective_permute
from neuronxcc.nki.typing import tensor
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark
from neuron_jax import nki_call
import neuronxcc.nki.nccl as nccl
import functools
import mx
import neuronxcc.nki.isa as nisa
import optax

def nki_quantization_matmul_jax(lhsT, rhs, result): #include k AND MULT DTYPE as param
  '''
  Kernel that fuses quantization with blocked matmul
  Load block -> calculate scale per block -> quantize and store block -> store scale -> blocked matmul
  '''
  # param shapes
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # tile sizes
  TILE_M = min(M, 128) 
  TILE_K = min(K, 128) 
  TILE_N = min(N, 128) 

  # initialize grids
  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]


  i_lhsT_scales = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs_scales = nl.mgrid[0:TILE_K, 0:TILE_N]

  i_scale1T = nl.mgrid[0:TILE_M, 0:TILE_N]
  i_scale2 = nl.mgrid[0:TILE_N, 0:TILE_M]

  # iterate through lhsT columns (e.g. for every TILE_M columns...)
  for m in nl.affine_range(M // TILE_M):
    # initialize scaled lhsT array
    lhsT_tiles_scaled = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_M),
                            dtype=nl.bfloat16, #CHANGE BFLOAT TO CUSTOM DTYPE
                            buffer=nl.sbuf) 

    # initialize lhsT scales array
    scale1T_tiles= nl.ndarray((K // TILE_K, nl.par_dim(TILE_M), TILE_N), dtype=nl.uint8,
                            buffer=nl.sbuf) 

    for k in nl.affine_range(K // TILE_K):
      # load slice of lhsT
      lhsT_slice = nl.load(lhsT[k * TILE_K + i_lhsT.p, m * TILE_M + i_lhsT.x])
      
      # initialize scale array (for this slice)
      scale = nl.ndarray((nl.par_dim(TILE_M), 1), dtype=nl.uint8,
                            buffer=nl.sbuf) 
    
      # calculate scale of lhsT_slice, store in scale array
      scale[...] = nl.subtract(nl.floor(nl.log(nl.max(nl.transpose(nl.abs(lhsT_slice)), 1))), 9)
      scale[...] = nl.power(nl.full(scale.shape, 2.0, lhsT.dtype), scale)
      scale1T_tiles[k] = scale.broadcast_to((TILE_M, TILE_N))

      #scale the lhsT_slice array using lhsT_scales
      lhsT_tiles_scaled[k] = nl.divide(lhsT_slice, nl.transpose(scale.broadcast_to((TILE_M, TILE_K))))

    for n in nl.affine_range(N // TILE_N):
      
      # initialize scaled lhsT array
      rhs_tiles_scaled = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                              dtype=nl.bfloat16, #CHANGE BFLOAT TO CUSTOM DTYPE
                              buffer=nl.sbuf) 

      # initialize lhsT scales array
      scale2_tiles= nl.ndarray((K // TILE_K, nl.par_dim(TILE_N), TILE_M), dtype=nl.uint8,
                              buffer=nl.sbuf) 

      for k in nl.affine_range(K // TILE_K):
        # load slice of lhsT
        rhs_slice =  nl.load(rhs[k * TILE_K + i_rhs.p, n * TILE_N + i_rhs.x])
        
        # initialize scale array (for this slice)
        scale2 = nl.ndarray((nl.par_dim(TILE_N), 1), dtype=nl.uint8,
                              buffer=nl.sbuf) 
      
    
        # calculate scale of lhsT_slice, store in scale array
        scale2[...] = nl.subtract(nl.floor(nl.log(nl.max(nl.transpose(nl.abs(rhs_slice)), 1))), 9)
        scale2[...] = nl.power(nl.full(scale2.shape, 2.0, rhs.dtype), scale2)
        scale2_tiles[k] = scale2.broadcast_to((TILE_N, TILE_M))
        
        #scale the lhsT_slice array using lhsT_scales
        rhs_tiles_scaled[k] = nl.divide(rhs_slice, nl.transpose(scale2.broadcast_to((TILE_N, TILE_K))))

      # Allocate a tile in PSUM for the result
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
      
        res_psum[...] += nl.multiply(nl.matmul(lhsT_tiles_scaled[k, i_lhsT.p, i_lhsT.x],
                                   rhs_tiles_scaled[k, i_rhs.p, i_rhs.x],
                                   transpose_x=True), nl.multiply(scale1T_tiles[k, i_scale1T.p, i_scale1T.x], nl.transpose(scale2_tiles[k, i_scale2.p, i_scale2.x])))

        # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sb)

@functools.partial(jax.jit, static_argnames=["k"])
def kernel_blocked_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int):
    result_shape = [*tensor1.shape[:-1], tensor2.shape[-1]]
    result = jnp.zeros(result_shape)
    if(len(tensor2.shape) > 2):
        for x in range(tensor2.shape[0]):
            new = kernel_blocked_matmul(tensor1[x], tensor2[x], k)
            result = result.at[x].set(new)
        return result 
    elif(len(tensor1.shape) > 2):
        for x in range(tensor1.shape[0]):
            result = result.at[x].set(kernel_blocked_matmul(tensor1[x], tensor2, k))
    if(len(tensor1.shape) <= 2 and len(tensor2.shape) <= 2):
      return kernel_matmul_wrap(tensor1, tensor2)
    return result

@jax.custom_gradient
def kernel_matmul_wrap(tensor1: jax.Array, tensor2: jax.Array):
  def grad(g):
    _, matmul_grad = jax.vjp(jnp.matmul, tensor1, tensor2)
    return matmul_grad(g)
  out_shape=jax.ShapeDtypeStruct((tensor1.shape[0], tensor2.shape[1]), dtype=tensor1.dtype)
  result = nki_call(nki_quantization_matmul_jax, tensor1.T, tensor2, out_shape=out_shape)
  return result, grad

#----------------------TESTS------------------------
@jax.jit
def namedscope_test(seq1, seq2):
  with jax.named_scope("tensormatmul"): 
    result = kernel_blocked_matmul(seq1, seq2, 128)

  return result

def profiletest():
  cpu_0 = jax.devices('cpu')[0]
  with jax.default_device(cpu_0):
    initializer = jax.nn.initializers.normal(0.01)
    seq1 = initializer(jax.random.PRNGKey(0), (256, 512), jnp.float32)
    seq2 = initializer(jax.random.PRNGKey(0), (512, 1024), jnp.float32)

  result = namedscope_test(seq1, seq2)

if __name__ == "__main__":
  profiletest()