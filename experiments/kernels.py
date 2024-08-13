import jax
from jax import numpy as jnp
import numpy
import neuronxcc.nki.language as nl
from neuronxcc.nki import tensor
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark
from neuron_jax import nki_call
import functools
import mx
# @jax_call

@baremetal(save_neff_name='file.neff', save_trace_name='profile.ntff')
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    result,
    # Meta-parameters
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=1,
    TILES_IN_BLOCK_K=4,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      result: the resulting output tensor of shape [M,N]
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  print(type(nl))
  TILE_M = nl.tile_size.gemm_xT_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_y_fmax  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K
  print(N)
  print(BLOCK_N)
  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype='bfloat16',
                            buffer=nl.sbuf)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because there is a sum reduction of the partial
    # sums over this loop, which introduces a loop-carried dependency
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                             dtype='bfloat16',
                             buffer=nl.sbuf)

      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                BLOCK_N * n + i_rhs.x], dtype='bfloat16')

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                dtype='bfloat16',
                                buffer=nl.sbuf)
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                   BLOCK_M * m + i_lhsT.x])

        # Do matmul with all tiles in the blocks
        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              res_tile[...] += nl.matmul(
                  lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x]) #* scale1[k][n] * scale2[k][m]

            # Reduce along contraction dimension due to the K dimension blocking
            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] = nl.loop_reduce(res_tile[i_res_mm.p,
                                                               i_res_mm.x],
                                                      op=jnp.add,
                                                      loop_indices=[k],
                                                      dtype='bfloat16')
    print("here")
    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
        result_packed = nl.ndarray((TILE_K, BLOCK_N),
                                   dtype='bfloat16',
                                   buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x])

# @baremetal(save_neff_name='file.neff', save_trace_name='profile.ntff')
def nki_matmul_fully_optimized_jax(
    lhsT,
    rhs,
    result,
    # Meta-parameters
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=1,
    TILES_IN_BLOCK_K=128,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      result: the resulting output tensor of shape [M,N]
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  print(type(nl))
  #TODO: VERIFY BELOW
  TILE_M = nl.tile_size.pmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.psum_fmax  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K
  print(M)
  print(BLOCK_M)
  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype='bfloat16',
                            buffer=nl.sbuf)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because there is a sum reduction of the partial
    # sums over this loop, which introduces a loop-carried dependency
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                             dtype='bfloat16',
                             buffer=nl.sbuf)

      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                BLOCK_N * n + i_rhs.x], dtype='bfloat16')

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                dtype='bfloat16',
                                buffer=nl.sbuf)
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                   BLOCK_M * m + i_lhsT.x])

        # Do matmul with all tiles in the blocks
        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              res_tile[...] += nl.matmul(
                  lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x]) #* scale1[k][n] * scale2[k][m]

            # Reduce along contraction dimension due to the K dimension blocking
            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] = nl.loop_reduce(res_tile[i_res_mm.p,
                                                               i_res_mm.x],
                                                      op=jnp.add,
                                                      loop_indices=[k],
                                                      dtype='bfloat16')
    print("here")
    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
        result_packed = nl.ndarray((TILE_K, BLOCK_N),
                                   dtype='bfloat16',
                                   buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x])       
@baremetal(save_neff_name='file.neff', save_trace_name='profile.ntff')
def nki_matmul_hoist_load_(lhsT, rhs, k, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner
     while hoisting the load of the lhsT and rhs to outer loops.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  TILE_M = 128 #nl.tile_size.gemm_xT_fmax  # 128
  TILE_K = k  # 128
  TILE_N = 512 #nl.tile_size.gemm_y_fmax  # 512

  # Define the indices (shape) of the tiles
  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_N numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    for k in nl.affine_range(K // TILE_K):
      # use `.p` for partition dimension and `.x` for the first free dimension
      lhsT_tiles[k, i_lhsT.p, i_lhsT.x] = nl.load(lhsT[k * TILE_K + i_lhsT.p,
                                                       m * TILE_M + i_lhsT.x])

    for n in nl.affine_range(N // TILE_N):

      # Load a whole column tiles from rhs (with K * TILE_M numbers)
      rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)
      for k in nl.affine_range(K // TILE_K):
        rhs_tiles[k, i_rhs.p, i_rhs.x] = nl.load(rhs[k * TILE_K + i_rhs.p,
                                                     n * TILE_N + i_rhs.x])

      # Allocate a tile in PSUM for the result
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # Accumulate partial-sums into PSUM
        res_psum[...] += nl.matmul(lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                                   rhs_tiles[k, i_rhs.p, i_rhs.x],
                                   transpose_x=True)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sb)

def nki_matmul_hoist_load_jax(lhsT, rhs, scale1T, scale2, k, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner
     while hoisting the load of the lhsT and rhs to outer loops.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  TILE_M = 128 #nl.tile_size.gemm_xT_fmax  # 128
  TILE_K = 128  # 128
  TILE_N = 512 #nl.tile_size.gemm_y_fmax  # 512

  # Define the indices (shape) of the tiles
  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
  #NEW vvv
  i_scale1T = nl.mgrid[0:TILE_M, 0:TILE_N]
  i_scale2 = nl.mgrid[0:TILE_M, 0:TILE_N]

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_N numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf) 
    scale1T_tiles = nl.ndarray((TILE_K, nl.par_dim(TILE_K), TILE_M),
                            dtype=scale1T.dtype,
                            buffer=nl.sbuf) 

    for k in nl.affine_range(K // TILE_K):
      # use `.p` for partition dimension and `.x` for the first free dimension
      lhsT_tiles[k, i_lhsT.p, i_lhsT.x] = nl.load(lhsT[k * TILE_K + i_lhsT.p,
                                                       m * TILE_M + i_lhsT.x])
      for scalek in nl.affine_range(TILE_K):
        scale1T_tiles[scalek, i_scale1T.p, i_scale1T.x] = nl.load(scale1T[k,
                                                       m * TILE_M + i_scale1T.x])

    for n in nl.affine_range(N // TILE_N):

      # Load a whole column tiles from rhs (with K * TILE_M numbers)
      rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)
      
      scale2_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                            dtype=scale2.dtype,
                            buffer=nl.sbuf)   

      for k in nl.affine_range(K // TILE_K):
        rhs_tiles[k, i_rhs.p, i_rhs.x] = nl.load(rhs[k * TILE_K + i_rhs.p,
                                                     n * TILE_N + i_rhs.x])
        scale2_tiles[k, i_scale2.p, i_scale2.x] = nl.load(scale2[k * TILE_K + i_scale2.p,
                                                       m * TILE_M + i_scale2.x])

      # Allocate a tile in PSUM for the result
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # #NEW vvv
        # scale1T_tile = nl.load(scale1T[k * 1,
        #                               m * TILE_M + i_scale1T.x])
        # scale2_tile = nl.load(scale2[k * 1,
        #                               n * TILE_N + i_scale2.x])

        # import pdb; pdb.set_trace()
        # Accumulate partial-sums into PSUM
        res_psum[...] += nl.multiply(nl.matmul(lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                                   rhs_tiles[k, i_rhs.p, i_rhs.x],
                                   transpose_x=True), nl.matmul(scale1T_tiles[k, i_scale1T.p, i_scale1T.x], scale2_tiles[k, i_scale2.p, i_scale2.x]))

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      jax.debug.print("{res_sb}", res_sb=res_sb)
      nl.store(result[m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sb)

def nki_quantization_matmul_jax(lhsT, rhs, result): #include k AND MULT DTYPE as param
  # param shapes
  K, M = lhsT.shape
  K_, N = rhs.shape
  # assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # tile sizes
  TILE_M = min(M, 128) #nl.tile_size.gemm_xT_fmax  # 128
  TILE_K = min(K, 128)  # 128
  TILE_N = min(N, 128) #nl.tile_size.gemm_y_fmax  # 512

  # initialize grids
  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]


  i_lhsT_scales = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs_scales = nl.mgrid[0:TILE_K, 0:TILE_N]

  i_scale1T = nl.mgrid[0:TILE_M, 0:TILE_N]
  i_scale2 = nl.mgrid[0:TILE_N, 0:TILE_M]
  # iterate through lhsT columns (e.g. for every TILE_M columns...)
  for m in nl.sequential_range(M // TILE_M):
    # initialize scaled lhsT array
    lhsT_tiles_scaled = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_M),
                            dtype=nl.float8_e4m3, #CHANGE BFLOAT TO CUSTOM DTYPE
                            buffer=nl.sbuf) 

    # initialize lhsT scales array
    scale1T_tiles= nl.ndarray((K // TILE_K, nl.par_dim(TILE_M), TILE_N), dtype=nl.uint8,
                            buffer=nl.sbuf) 

    for k in nl.affine_range(K // TILE_K):
      # lhsT_slice = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype,
      #                       buffer=nl.sbuf) 
      # load slice of lhsT
      lhsT_slice = nl.load(lhsT[k * TILE_K + i_lhsT.p, m * TILE_M + i_lhsT.x])
      
      # initialize scale array (for this slice)
      scale = nl.ndarray((nl.par_dim(TILE_M), 1), dtype=nl.uint8,
                            buffer=nl.sbuf) 
      
      # initialize scales (broadcasted to the size of lhsT_tiles)
      # lhsT_scales = nl.ndarray((nl.par_dim(TILE_M), TILE_K),
      #                       dtype=nl.uint8,
      #                       buffer=nl.sbuf)

      # calculate scale of lhsT_slice, store in scale array
      shared_exp = nl.subtract(nl.floor(nl.log(nl.max(nl.transpose(nl.abs(lhsT_slice)), 1))), 9)
      scale = nl.minimum(nl.maximum(nl.power(nl.full(shared_exp.shape, 2.0, lhsT.dtype), shared_exp), 1), 127).astype(nl.uint8)
      scale1T_tiles[k] = scale.broadcast_to((TILE_M, TILE_N))
      # broadcasting scales to necessary size (broadcast function not available)
      # for i_n in nl.affine_range(TILE_N):
      #   scale1T_tiles[k, 0:TILE_M, i_n] = scale 

      # for i_k in nl.affine_range(TILE_K):
      #   lhsT_scales[0:TILE_M, i_k] = scale 

      #scale the lhsT_slice array using lhsT_scales
      lhsT_tiles_scaled[k] = nl.divide(lhsT_slice, nl.transpose(scale.broadcast_to((TILE_M, TILE_K))))

    for n in nl.affine_range(N // TILE_N):
      
      # initialize scaled lhsT array
      rhs_tiles_scaled = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                              dtype=nl.float8_e4m3, #CHANGE BFLOAT TO CUSTOM DTYPE
                              buffer=nl.sbuf) 

      # initialize lhsT scales array
      scale2_tiles= nl.ndarray((K // TILE_K, nl.par_dim(TILE_N), TILE_M), dtype=nl.uint8,
                              buffer=nl.sbuf) 

      for k in nl.affine_range(K // TILE_K):
        # rhs_slice = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype,
        #                     buffer=nl.sbuf) 
        # load slice of lhsT
        rhs_slice =  nl.load(rhs[k * TILE_K + i_rhs.p, n * TILE_N + i_rhs.x])
        
        # initialize scale array (for this slice)
        scale = nl.ndarray((nl.par_dim(TILE_N), 1), dtype=nl.uint8,
                              buffer=nl.sbuf) 
          
        # # initialize scales (broadcasted to the size of lhsT_tiles)
        # rhs_scales = nl.ndarray((nl.par_dim(TILE_N), TILE_K),
        #                       dtype=nl.uint8,
        #                       buffer=nl.sbuf)

        # calculate scale of lhsT_slice, store in scale array
        shared_exp = nl.subtract(nl.floor(nl.log(nl.max(nl.transpose(nl.abs(rhs_slice)), 1))), 9)
        scale= nl.minimum(nl.maximum(nl.power(nl.full(shared_exp.shape, 2.0, rhs.dtype), shared_exp), 1), 127).astype(nl.uint8)
        
        scale2_tiles[k] = scale.broadcast_to((TILE_N, TILE_M)) #TRANSPOSE BEFORE BROADCAST
        # scale2_tiles[k, 0:TILE_N, 0:TILE_M] = scale.broadcast_to((TILE_N, TILE_M))
        # broadcasting scales to necessary size (broadcast function not available)
        # for i_m in nl.affine_range(TILE_M):
        #   scale2_tiles[k, 0:TILE_N, i_m] = scale 

        # for i_k in nl.affine_range(TILE_K):
        #   rhs_scales[0:TILE_N, i_k] = scale 

        #scale the lhsT_slice array using lhsT_scales
        rhs_tiles_scaled[k] = nl.divide(rhs_slice, nl.transpose(scale.broadcast_to((TILE_N, TILE_K))))

      # Allocate a tile in PSUM for the result
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
      
        res_psum[...] += nl.multiply(nl.matmul(lhsT_tiles_scaled[k, i_lhsT.p, i_lhsT.x],
                                   rhs_tiles_scaled[k, i_rhs.p, i_rhs.x],
                                   transpose_x=True), nl.multiply(scale1T_tiles[k, i_scale1T.p, i_scale1T.x], nl.transpose(scale2_tiles[k, i_scale2.p, i_scale2.x])))

        # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sb)

def nki_reg_matmul_jax(lhsT, rhs, result): #include k AND MULT DTYPE as param
  # param shapes
  K, M = lhsT.shape
  K_, N = rhs.shape
  # assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # tile sizes
  TILE_M = 128 #nl.tile_size.gemm_xT_fmax  # 128
  TILE_K = 128  # 128
  TILE_N = 128 #nl.tile_size.gemm_y_fmax  # 512

  # initialize grids
  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]


  i_lhsT_scales = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs_scales = nl.mgrid[0:TILE_K, 0:TILE_N]

  i_scale1T = nl.mgrid[0:TILE_M, 0:TILE_N]
  i_scale2 = nl.mgrid[0:TILE_M, 0:TILE_N]
  # iterate through lhsT columns (e.g. for every TILE_M columns...)
  for m in nl.sequential_range(M // TILE_M):
    # initialize scaled lhsT array
    lhsT_tiles_scaled = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_M),
                            dtype=nl.float8_e4m3, #CHANGE BFLOAT TO CUSTOM DTYPE
                            buffer=nl.sbuf) 

    # initialize lhsT scales array
    scale1T_tiles= nl.ndarray((K // TILE_K, nl.par_dim(TILE_M), TILE_N), dtype=nl.uint8,
                            buffer=nl.sbuf) 

    for k in nl.affine_range(K // TILE_K):
      # load slice of lhsT
      lhsT_slice =  nl.load(lhsT[k * TILE_K + i_lhsT.p, m * TILE_M + i_lhsT.x])
      
      # initialize scale array (for this slice)
      scale = nl.ndarray((nl.par_dim(TILE_K), 1), dtype=nl.uint8,
                            buffer=nl.sbuf) 
      
      # initialize scales (broadcasted to the size of lhsT_tiles)
      lhsT_scales = nl.ndarray((nl.par_dim(TILE_M), TILE_K),
                            dtype=nl.uint8,
                            buffer=nl.sbuf)

      # calculate scale of lhsT_slice, store in scale array
      shared_exp = nl.subtract(nl.floor(nl.log(nl.max(nl.transpose(nl.abs(lhsT_slice)), 1))), 9)
      scale[...] = nl.minimum(nl.maximum(nl.power(nl.full(shared_exp.shape, 2.0, lhsT.dtype), shared_exp), 1), 127)

      # broadcasting scales to necessary size (broadcast function not available)
      for i_n in nl.affine_range(TILE_N):
        scale1T_tiles[k, 0:TILE_M, i_n] = scale 

      for i_k in nl.affine_range(TILE_K):
        lhsT_scales[0:TILE_M, i_k] = scale 

      #scale the lhsT_slice array using lhsT_scales
      lhsT_tiles_scaled[k, i_lhsT.p, i_lhsT.x] = nl.divide(lhsT_slice, nl.transpose(lhsT_scales))

    for n in nl.affine_range(N // TILE_N):
      
      # initialize scaled lhsT array
      rhs_tiles_scaled = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                              dtype=nl.float8_e4m3, #CHANGE BFLOAT TO CUSTOM DTYPE
                              buffer=nl.sbuf) 

      # initialize lhsT scales array
      scale2_tiles= nl.ndarray((K // TILE_K, nl.par_dim(TILE_N), TILE_M), dtype=nl.uint8,
                              buffer=nl.sbuf) 

      for k in nl.affine_range(K // TILE_K):
        # load slice of lhsT
        rhs_slice =  nl.load(rhs[k * TILE_K + i_rhs.p, n * TILE_N + i_rhs.x])
        
        # initialize scale array (for this slice)
        scale = nl.ndarray((nl.par_dim(TILE_K), 1), dtype=nl.uint8,
                              buffer=nl.sbuf) 
          
        # initialize scales (broadcasted to the size of lhsT_tiles)
        rhs_scales = nl.ndarray((nl.par_dim(TILE_N), TILE_K),
                              dtype=nl.uint8,
                              buffer=nl.sbuf)

        # calculate scale of lhsT_slice, store in scale array
        shared_exp = nl.subtract(nl.floor(nl.log(nl.max(nl.transpose(nl.abs(rhs_slice)), 1))), 9)
        scale[...] = nl.minimum(nl.maximum(nl.power(nl.full(shared_exp.shape, 2.0, rhs.dtype), shared_exp), 1), 127)

        # broadcasting scales to necessary size (broadcast function not available)
        for i_m in nl.affine_range(TILE_M):
          scale2_tiles[k, 0:TILE_N, i_m] = scale 

        for i_k in nl.affine_range(TILE_K):
          rhs_scales[0:TILE_N, i_k] = scale 

        #scale the lhsT_slice array using lhsT_scales
        rhs_tiles_scaled[k, i_rhs.p, i_rhs.x] = nl.divide(rhs_slice, nl.transpose(rhs_scales))

      # Allocate a tile in PSUM for the result
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
      
        res_psum[...] += nl.multiply(nl.matmul(lhsT_tiles_scaled[k, i_lhsT.p, i_lhsT.x],
                                   rhs_tiles_scaled[k, i_rhs.p, i_rhs.x],
                                   transpose_x=True), nl.multiply(nl.transpose(scale1T_tiles[k, i_scale1T.p, i_scale1T.x]), scale2_tiles[k, i_scale2.p, i_scale2.x]))

        # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sb)
# @jax.jit
def nki_quantization_matmul_jax_wrapper(lhs, rhs):
  out_shape=jax.ShapeDtypeStruct((lhs.shape[0], 512), dtype=lhs.dtype)
  out_shape = nki_call(nki_quantization_matmul_jax, lhs.T, rhs, out_shape=out_shape)
  return out_shape


def test():
  #initialize params
    d_model = 512
    d_ff = 1024
    batch_size = 32
    sequence_length = 16

    
    # seq1 = numpy.random.rand(256, 512)
    # seq2 = numpy.random.rand(512, 1024)
    # result = numpy.zeros([256, 1024], dtype='bfloat16')
    #     #initialize params
    prng_key = jax.random.PRNGKey(0)
    #initialize seq
    cpu_0 = jax.devices('cpu')[0]
    with jax.default_device(cpu_0):
        initializer = jax.nn.initializers.normal(10)
        seq1 = initializer(prng_key, (256, 512), jnp.float32)
        seq2 = initializer(prng_key, (512, 1024), jnp.float32)
        # scale1 = jnp.ones((128, 2), jnp.uint8) #REDO WITH INTTTT
        # scale2 = jnp.ones((2, 512), jnp.uint8)
        # print(seq1)

    # nki_call(nki_matmul_fully_optimized_jax, seq1.T, seq2, out_shape=jax.ShapeDtypeStruct((seq1.shape[0], seq2.shape[1]), dtype=seq1.dtype))
    # print(result)
    # with jax.named_scope("KERNEL"):
    #   result = nki_call(nki_matmul_hoist_load_jax, seq1.T, seq2, scale1.T, scale2, 128, out_shape=jax.ShapeDtypeStruct((seq1.shape[0], seq2.shape[1]), dtype=seq1.dtype))
    # with jax.named_scope("TWOD MATMUL"):
    #   result = mx.twod(seq1, seq2, scale1, scale2, 128)
    print(jnp.matmul(seq1, seq2))
    out_shape=jax.ShapeDtypeStruct((seq1.shape[0], seq2.shape[1]), dtype=seq1.dtype)
    # import pdb; pdb.set_trace()
    result = nki_call(nki_quantization_matmul_jax, seq1.T, seq2, out_shape=out_shape)
    # result = nki_quantization_matmul_jax_wrapper(seq1, seq2)
    print(result)
    # print(result.dtype)

@functools.partial(jax.jit, static_argnames=["k"])
def kernel_blocked_matmul(tensor1: jax.Array, tensor2: jax.Array, k: int):
    '''
    Blocked matrix multiplication for n-d arrays
    - assume tensor1 + scale1 n-dim, tensor2 + scale2 2-dim
    '''
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
        out_shape=jax.ShapeDtypeStruct((tensor1.shape[0], tensor2.shape[1]), dtype=tensor1.dtype)
        result = nki_call(nki_quantization_matmul_jax, tensor1.T, tensor2, out_shape=out_shape)
        return result
    return result

@jax.jit
def namedscope(seq1, seq2):
  with jax.named_scope("KERNEL"): 
    result = blocked_matmul(seq1, seq2, 128)

  return result

def profiletest():
  cpu_0 = jax.devices('cpu')[0]
  with jax.default_device(cpu_0):
    initializer = jax.nn.initializers.normal(0.01)
    seq1 = initializer(jax.random.PRNGKey(0), (256, 512), jnp.float32)
    seq2 = initializer(jax.random.PRNGKey(0), (512, 1024), jnp.float32)

  result = namedscope(seq1, seq2)
  print(result)
if __name__ == "__main__":
  profiletest()