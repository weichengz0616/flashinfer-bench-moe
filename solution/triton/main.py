import torch
from torch.utils.cpp_extension import load, load_inline
import time

import torch.utils.cpp_extension

import triton
import triton.language as tl
from typing import List, Sequence
from dataclasses import dataclass


#################################################################################
#################################################################################
#################################################################################
#################################################################################
@dataclass
class GemmConfig:
    block_size_m: int
    block_size_n: int
    block_size_k: int
    num_stages: int
    use_tma: bool


def get_blk_size_m(args):
    if args["seq_len"] <= 100:
        return 16
    elif args["seq_len"] <= 500:
        return 32
    elif args["seq_len"] <= 2000:
        return 64
    elif args["seq_len"] <= 20000:
        return 128
    else:
        return 256
# @triton.heuristics({
#     "BLOCK_SIZE_M": get_blk_size_m,
# })
@triton.jit
def gemm1_kernel(
    # input
    a_base_ptr, # [sum(s_i), 7168], fp8
    a_scale_base_ptr, # [7168//128, sum(s_i)], fp32
    a_offset_ptr, # [33], int32
    b_base_ptr, # [32, 4096, 7168], fp8
    b_scale_base_ptr, # [32, 4096//128, 7168//128], fp32
    seq_len,
    # output
    c_base_ptr, # [sum(s_i), 4096], fp32
    c_scale_base_ptr, # [2048//128, sum(s_i)], fp32
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    # num_valid_tokens = tl.load(a_offset_ptr + 32) # sum(s_i)
    # 遍历 gemm
    for i in range(32):
        offset = tl.load(a_offset_ptr + i)
        gm = tl.load(a_offset_ptr + i + 1) - offset # s_i
        gn = 4096
        gk = 7168
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N) # 处理两个 tile
        num_tiles = (num_m_tiles * num_n_tiles).to(tl.int32)

        # 检查当前 tile 编号是否还在当前 gemm 范围内
        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda = 7168
            ldb = 7168
            ldc = 4096

            a_ptr = a_base_ptr + offset * lda
            b_ptr = b_base_ptr + gn * gk * i
            c_ptr = c_base_ptr + offset * ldc

            a_scale_ptr = a_scale_base_ptr + offset
            b_scale_ptr = b_scale_base_ptr + gn // 128 * gk // 128 * i
            # c_scale_ptr = c_scale_base_ptr + tl.load(a_offset_ptr + i)

            # TMA
            if USE_TMA:
                a_desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[gm, gk],
                    strides=[lda, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            # b2_desc = tl.make_tensor_descriptor(
            #     b_ptr,
            #     shape=[gn, gk],
            #     strides=[ldb, 1],
            #     block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            # )
            if USE_TMA:
                c_desc = tl.make_tensor_descriptor(
                    c_ptr,
                    shape=[gm, gn],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # 在当前 gemm 内拿多个 tile
            while (tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles):
                k = gk
                tile_idx_in_gemm = tile_idx - last_gemm_end_tile_idx
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn1 = tile_n_idx * BLOCK_SIZE_N
                # offs_bn2 = (tile_n_idx + num_n_tiles) * BLOCK_SIZE_N

                # 缩放矩阵每一行的元素个数
                num_k_blocks = tl.cdiv(k, 128)

                acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                # acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in tl.range(0, tl.cdiv(k, BLOCK_SIZE_K), num_stages=6, warp_specialize=True):
                    if USE_TMA:
                        a = a_desc.load(
                            [offs_am, kk * BLOCK_SIZE_K],
                        )
                    else:
                        a_idx = a_ptr + offs_am * lda + kk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_M)[:, None] * lda + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        a = tl.load(a_idx, mask=((offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None]) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    b1 = b1_desc.load(
                        [offs_bn1, kk * BLOCK_SIZE_K],
                    )
                    # b2 = b2_desc.load(
                    #     [offs_bn2, kk * BLOCK_SIZE_K],
                    # )
                    
                    # b1 = tl.load(b1_idx) # [BLOCK_SIZE_N, BLOCK_SIZE_K]
                    # b2 = tl.load(b2_idx) # [BLOCK_SIZE_N, BLOCK_SIZE_K
                    
                    # 获取当前 128x128 block 的缩放因子
                    a_scale_idx = a_scale_ptr + offs_am + kk * seq_len * 8 + tl.arange(0, BLOCK_SIZE_M)
                    a_scale_vec = tl.load(a_scale_idx, mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm, other=1.0)
                    a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1)) # 广播到 [M, 1]
                    # b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128)) # 1 个数
                    # acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    if BLOCK_SIZE_N == 128:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128)) # 1 个数
                        acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    else:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        b1_scale2 = tl.load(b_scale_ptr + ((offs_bn1 + 128) // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        scales = tl.join(b1_scale, b1_scale2)
                        b_scale_vec = tl.reshape(scales, (2, 1))
                        scales_expanded = tl.broadcast_to(b_scale_vec, (2, 128))
                        final_scales = tl.reshape(scales_expanded, (1, 256))
                        acc1 += tl.dot(a, b1.T) * a_scale_col * final_scales
                    # acc2 += tl.dot(a, b2.T) * a_scale_col * b2_scale
                    # a = a.to(tl.float32) * a_scale_col
                    # b1 = b1.to(tl.float32) * b1_scale
                    # b2 = b2.to(tl.float32) * b2_scale
                    # acc1 += tl.dot(a, b1.T, input_precision="ieee")
                    # acc2 += tl.dot(a, b2.T, input_precision="ieee")
                    

                # activation
                # silu_x2 = acc2 / (1.0 + tl.exp(-acc2))
                # res_c = silu_x2 * acc1

                # if tile_idx == 0:
                #     tl.device_print("Partial Block:", res_c, acc1, acc2)

                # 量化
                # abs_c = tl.abs(res_c)
                # 在 N 维度上做 reduce，找最大值
                # max_val 的 shape 为 [BLOCK_SIZE_M]
                # max_val = tl.max(abs_c, axis=1)
                # fp8_max = 448.0
                # tile_scale = max_val / fp8_max
                # res_c = (res_c / tl.reshape(tile_scale, (BLOCK_SIZE_M, 1))).to(tl.float8e4nv)

                # 写回
                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N
                if USE_TMA:
                    c_desc.store(
                        [offs_cm, offs_cn],
                        acc1.to(tl.float16),
                    )
                else:
                    c_idx = c_ptr + offs_cm * ldc + offs_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * ldc + tl.arange(0, BLOCK_SIZE_N)[None, :]
                    tl.store(c_idx, acc1.to(tl.bfloat16), mask=((offs_cm + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None])
                # c_scale_idx = c_scale_ptr + offs_cm + tile_n_idx * seq_len * 8 + tl.arange(0, BLOCK_SIZE_M)
                # tl.store(c_scale_idx, tile_scale, mask=(offs_cm + tl.arange(0, BLOCK_SIZE_M)) < gm)


                # 下一个 tile
                tile_idx += NUM_SM

        # 进入下一个 gemm
        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles

def launch_gemm1_kernel(
    permute_hidden_states: torch.tensor, # [sum(s), 7168], fp8
    permute_hidden_states_scale: torch.tensor, # [7168//128, sum(s)], fp32
    offset: torch.tensor, # [33], int32, 每个 expert 处理的行数的前缀和，最后一个元素是 sum(s)
    gemm1_weights: torch.tensor, # [32, 4096, 7168], fp8
    gemm1_weights_scale: torch.tensor, # [32, 4096//128, 7168//128], fp32
    seq_len,
    output: torch.tensor, # [sum(s), 2048], fp32
    output_scale: torch.Tensor, # [2048//128, sum(s)], fp32
    *,
    block_size_m: int = 32,
    block_size_n: int = 128,
    block_size_k: int = 128,
    num_sm: int = 160,
) -> List[torch.Tensor]:
    # group_size = len(a_list)
    # device = permute_hidden_states.device

    # num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (num_sm,)

    gemm1_kernel[grid](
        # input
        a_base_ptr=permute_hidden_states,
        a_scale_base_ptr=permute_hidden_states_scale,
        a_offset_ptr=offset,
        b_base_ptr=gemm1_weights,
        b_scale_base_ptr=gemm1_weights_scale,
        seq_len=seq_len,
        # output
        c_base_ptr=output,
        c_scale_base_ptr=output_scale,
        NUM_SM=num_sm,
        # BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        num_stages=4,
        num_warps=8,
    )
    return


# input:
# [sum(s), 7168]
# -> 共 32 个 expert，sum(s) 可以分成 32 部分，每个部分的长度，以及每个部分的 seq id，即每个 expert 实际上是处理的哪些行
def gemm1(
    permute_hidden_states,
    permute_hidden_states_scale,
    offset,
    seq_len,
    gemm1_weights,
    gemm1_weights_scale,
    output,
    output_scale,
    num_sm=160,
): # return [sum(s), 2048]
    
    # output = torch.empty((permute_hidden_states.shape[0], 2048), device=permute_hidden_states.device, dtype=torch.float32)

    launch_gemm1_kernel(
        permute_hidden_states,
        permute_hidden_states_scale,
        offset,
        gemm1_weights,
        gemm1_weights_scale,
        seq_len,
        output,
        output_scale,
        block_size_m=32,
        block_size_n=128,
        block_size_k=128,
        num_sm=num_sm,
    )

    return



from triton.compiler import ASTSource, compile
class gemm1Kernel:
    def __init__(self, num_sm):
        signature = {
            'a_base_ptr': '*fp8e4nv',
            'a_scale_base_ptr': '*fp32',
            'a_offset_ptr': '*i32',
            'b_base_ptr': '*fp8e4nv',
            'b_scale_base_ptr': '*fp32',
            'seq_len': 'i32',
            'c_base_ptr': '*fp16',
            'c_scale_base_ptr': '*fp32',
            'NUM_SM': 'constexpr',
            'BLOCK_SIZE_M': 'constexpr',
            'BLOCK_SIZE_N': 'constexpr',
            'BLOCK_SIZE_K': 'constexpr',
            'USE_TMA': 'constexpr',
        }

        # blk_sizes = [64, 128]
        # num_stages = [8, 6]
        self.configs = []
        self.configs.append(GemmConfig(block_size_m=64, block_size_n=128, block_size_k=128, num_stages=8, use_tma=False))
        self.configs.append(GemmConfig(block_size_m=64, block_size_n=256, block_size_k=128, num_stages=4, use_tma=False))
        self.configs.append(GemmConfig(block_size_m=128, block_size_n=128, block_size_k=128, num_stages=6, use_tma=True))
        constexprs_list = []
        options_list = []
        for i in range(len(self.configs)):
            constexprs_list.append({
                (8,): num_sm, 
                (9,): self.configs[i].block_size_m, 
                (10,): self.configs[i].block_size_n, 
                (11,): self.configs[i].block_size_k,
                (12,): self.configs[i].use_tma
            })
            options_list.append({
                "num_warps": 8,
                "num_stages": self.configs[i].num_stages,
            })
        
        attrs = {
            (0,): [['tt.divisibility', 16]],
            (1,): [['tt.divisibility', 16]],
            (2,): [['tt.divisibility', 16]],
            (3,): [['tt.divisibility', 16]],
            (4,): [['tt.divisibility', 16]],
            (5,): [['tt.divisibility', 16]],
            (6,): [['tt.divisibility', 16]],
            (7,): [['tt.divisibility', 16]]
        }

        self.num_sm = num_sm
        self.kernels = []
        for i in range(len(self.configs)):
            src = ASTSource(
                fn=gemm1_kernel,
                signature=signature,
                constexprs=constexprs_list[i],
                attrs=attrs
            )
            compiled_kernel = compile(src, options=options_list[i])
            # compiled_kernel = compile(src)
            self.kernels.append(compiled_kernel)

    def __call__(
            self, 
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            seq_len,
            c_base,
            c_scale_base, 
            stream=None):
        if stream is None:
            device = triton.runtime.driver.active.get_current_device()
            stream = triton.runtime.driver.active.get_current_stream(device)
        elif hasattr(stream, "cuda_stream"):
            # 兼容传入 torch stream 对象
            stream = stream.cuda_stream

        kernel = None
        config = None
        if seq_len <= 4:
            kernel = self.kernels[0]
            config = self.configs[0]
        elif seq_len <= 1000:
            kernel = self.kernels[1]
            config = self.configs[1]
        else:
            kernel = self.kernels[2]
            config = self.configs[2]

        grid = (self.num_sm, 1, 1)
        launch_metadata = kernel.launch_metadata(grid, stream, a_base, a_scale_base, a_offset, b_base, b_scale_base, seq_len, c_base, c_scale_base)

        kernel.run(
            grid[0], grid[1], grid[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            None,
            None,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            seq_len,
            c_base,
            c_scale_base,
            self.num_sm,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.use_tma
        )
        return


#################################################################################
#################################################################################
#################################################################################
#################################################################################


# @triton.heuristics({
#     "BLOCK_SIZE_M": get_blk_size_m,
# })
@triton.jit
def gemm2_kernel(
    # input
    a_base_ptr,
    a_scale_base_ptr, # [2048//128, sum(s_i)], fp32
    a_offset_ptr,
    b_base_ptr,
    b_scale_base_ptr,
    permute_weights_base_ptr, # [group_size], fp32 -> [s_i]
    permute_token_idx_base_ptr, # [group_size], int32 -> [s_i]
    seq_len,
    #output
    output_ptr, # [sum(s_i), 7168], fp32
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    # num_valid_tokens = tl.load(a_offset_ptr + 32)
    # 遍历 gemm
    for i in range(32):
        # get the gemm size of the current problem
        offset = tl.load(a_offset_ptr + i)
        gm = tl.load(a_offset_ptr + i + 1) - offset
        gn = 7168
        gk = 2048
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # 检查当前 tile 编号是否还在当前 gemm 范围内
        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda = 2048
            ldb = 2048
            ldc = 7168

            a_ptr = a_base_ptr + offset * lda
            b_ptr = b_base_ptr + gn * gk * i
            c_ptr = output_ptr + offset * ldc
            
            a_scale_ptr = a_scale_base_ptr + offset
            b_scale_ptr = b_scale_base_ptr + gn // 128 * gk // 128 * i

            # 当前 expert 处理的 tokens 对应的 weight，长度为 gm
            p_weight_ptr = permute_weights_base_ptr + offset
            # 当前 expert 处理的 tokens 对应的 token idx，长度为 gm，决定写回 output 的位置
            p_source_idx_ptr = permute_token_idx_base_ptr + offset

            # TMA
            if USE_TMA:
                a_desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[gm, gk],
                    strides=[lda, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            if USE_TMA:
                c_desc = tl.make_tensor_descriptor(
                    c_ptr,
                    shape=[gm, gn],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # 在当前 gemm 内拿多个 tile
            while (tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles):
                k = gk
                tile_idx_in_gemm = tile_idx - last_gemm_end_tile_idx
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn1 = tile_n_idx * BLOCK_SIZE_N

                # 缩放矩阵每一行的元素个数
                num_k_blocks = tl.cdiv(k, 128)
                acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in tl.range(0, tl.cdiv(k, BLOCK_SIZE_K), num_stages=8, warp_specialize=True):
                    if USE_TMA:
                        a = a_desc.load(
                            [offs_am, kk * BLOCK_SIZE_K],
                        )
                    else:
                        a_idx = a_ptr + offs_am * lda + kk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_M)[:, None] * lda + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        a = tl.load(a_idx, mask=((offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None]) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    b1 = b1_desc.load(
                        [offs_bn1, kk * BLOCK_SIZE_K],
                    )

                    a_scale_idx = a_scale_ptr + offs_am + kk * seq_len * 8 + tl.arange(0, BLOCK_SIZE_M)
                    a_scale_vec = tl.load(a_scale_idx, mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm, other=1.0)
                    a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1)) # 广播到 [M, 1]
                    if BLOCK_SIZE_N == 128:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128)) # 1 个数
                        acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    else:
                        b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        b1_scale2 = tl.load(b_scale_ptr + ((offs_bn1 + 128) // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                        scales = tl.join(b1_scale, b1_scale2)
                        b_scale_vec = tl.reshape(scales, (2, 1))
                        scales_expanded = tl.broadcast_to(b_scale_vec, (2, 128))
                        final_scales = tl.reshape(scales_expanded, (1, 256))
                        acc1 += tl.dot(a, b1.T) * a_scale_col * final_scales

                # 和 weight 相乘
                offs_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offs_m < gm
                tile_weights = tl.load(p_weight_ptr + offs_m, mask=mask_m, other=0.0)
                acc1 = acc1 * tl.reshape(tile_weights, (BLOCK_SIZE_M, 1))

                # 写回 output: scatter-add
                # source_indices = tl.load(p_source_idx_ptr + offs_m, mask=mask_m, other=0)
                # offs_n = offs_bn1 + tl.arange(0, BLOCK_SIZE_N)
                # mask_n = offs_n < gn
                # source_indices_col = tl.reshape(source_indices, (BLOCK_SIZE_M, 1))
                # offs_n_row = tl.reshape(offs_n, (1, BLOCK_SIZE_N))
                # mask_m_col = tl.reshape(mask_m, (BLOCK_SIZE_M, 1))
                # mask_n_row = tl.reshape(mask_n, (1, BLOCK_SIZE_N))
                # dest_offsets = source_indices_col * 7168 + offs_n_row
                # full_mask = mask_m_col & mask_n_row
                # tl.atomic_add(output_ptr + dest_offsets, acc1, mask=full_mask)

                # 写回连续缓冲区
                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N
                if USE_TMA:
                    c_desc.store(
                        [offs_cm, offs_cn],
                        acc1.to(tl.bfloat16),
                    )
                else:
                    c_idx = c_ptr + offs_cm * ldc + offs_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * ldc + tl.arange(0, BLOCK_SIZE_N)[None, :]
                    tl.store(c_idx, acc1.to(tl.bfloat16), mask=((offs_cm + tl.arange(0, BLOCK_SIZE_M)) < gm)[:, None])

                # 下一个 tile
                tile_idx += NUM_SM

        # 进入下一个 gemm
        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles


def launch_gemm2_kernel(
    gemm1_output,
    gemm1_output_scale, # [2048//128, sum(s_i)], fp32
    offset,
    gemm2_weights,
    gemm2_weights_scale,
    permute_weights,
    permute_token_idx,
    seq_len: int,
    output: torch.Tensor,
    *,
    block_size_m: int = 32,
    block_size_n: int = 128,
    block_size_k: int = 128,
    num_sm: int = 160,
) -> torch.Tensor:
    # if block_size_k % 128 != 0:
    #     raise ValueError("block_size_k must be a multiple of 128 for current scale indexing")

    # device = output.device

    # num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (num_sm,)

    gemm2_kernel[grid](
        gemm1_output,
        gemm1_output_scale,
        offset,
        gemm2_weights,
        gemm2_weights_scale,
        permute_weights,
        permute_token_idx,
        seq_len,
        output,
        NUM_SM=num_sm,
        # BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        num_stages=4,
        num_warps=8,
    )
    return output


def gemm2(
    gemm1_output,
    gemm1_output_scale, # [2048//128, sum(s_i)], fp32
    offset,
    gemm2_weights,
    gemm2_weights_scale,
    permute_weights,
    permute_token_idx,
    seq_len: int,
    output,
    num_sm=160,
):
        
    # output = torch.zeros((seq_len, 7168), device=gemm1_output.device, dtype=torch.float32)

    launch_gemm2_kernel(
        gemm1_output,
        gemm1_output_scale,
        offset,
        gemm2_weights,
        gemm2_weights_scale,
        permute_weights,
        permute_token_idx,
        seq_len=seq_len,
        output=output,
        block_size_m=32,
        block_size_n=128,
        block_size_k=128,
        num_sm=num_sm,
    )

    return output



class gemm2Kernel:
    def __init__(self, num_sm):
        signature = {
            'a_base_ptr': '*fp8e4nv',
            'a_scale_base_ptr': '*fp32',
            'a_offset_ptr': '*i32',
            'b_base_ptr': '*fp8e4nv',
            'b_scale_base_ptr': '*fp32',
            'permute_weights_base_ptr': '*fp32',
            'permute_token_idx_base_ptr': '*i32',
            'seq_len': 'i32',
            'output_ptr': '*bf16',
            'NUM_SM': 'constexpr',
            'BLOCK_SIZE_M': 'constexpr',
            'BLOCK_SIZE_N': 'constexpr',
            'BLOCK_SIZE_K': 'constexpr',
            'USE_TMA': 'constexpr',
        }

        self.configs = []
        self.configs.append(GemmConfig(block_size_m=64, block_size_n=256, block_size_k=128, num_stages=4, use_tma=False))
        self.configs.append(GemmConfig(block_size_m=128, block_size_n=128, block_size_k=128, num_stages=4, use_tma=True))
        constexprs_list = []
        options_list = []
        for i in range(len(self.configs)):
            constexprs_list.append({
                (9,): num_sm, 
                (10,): self.configs[i].block_size_m, 
                (11,): self.configs[i].block_size_n, 
                (12,): self.configs[i].block_size_k,
                (13,): self.configs[i].use_tma
            })
            options_list.append({
                "num_warps": 8,
                "num_stages": self.configs[i].num_stages,
            })
        
        attrs = {
            (0,): [['tt.divisibility', 16]],
            (1,): [['tt.divisibility', 16]],
            (2,): [['tt.divisibility', 16]],
            (3,): [['tt.divisibility', 16]],
            (4,): [['tt.divisibility', 16]],
            (5,): [['tt.divisibility', 16]],
            (6,): [['tt.divisibility', 16]],
            (7,): [['tt.divisibility', 16]],
            (8,): [['tt.divisibility', 16]]
        }

        self.num_sm = num_sm
        self.kernels = []
        for i in range(len(self.configs)):
            src = ASTSource(
                fn=gemm2_kernel,
                signature=signature,
                constexprs=constexprs_list[i],
                attrs=attrs
            )
            compiled_kernel = compile(src, options=options_list[i])
            # compiled_kernel = compile(src)
            self.kernels.append(compiled_kernel)


    def __call__(
            self, 
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            permute_weights,
            permute_token_idx,
            seq_len,
            output,
            stream=None):
        if stream is None:
            device = triton.runtime.driver.active.get_current_device()
            stream = triton.runtime.driver.active.get_current_stream(device)
        elif hasattr(stream, "cuda_stream"):
            # 兼容传入 torch stream 对象
            stream = stream.cuda_stream

        kernel = None
        config = None
        if seq_len <= 1000:
            kernel = self.kernels[0]
            config = self.configs[0]
        else:
            kernel = self.kernels[1]
            config = self.configs[1]

        grid = (self.num_sm, 1, 1)
        launch_metadata = kernel.launch_metadata(grid, stream, a_base, a_scale_base, a_offset, b_base, b_scale_base, permute_weights, permute_token_idx, seq_len, output)

        kernel.run(
            grid[0], grid[1], grid[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            None,
            None,
            a_base,
            a_scale_base,
            a_offset,
            b_base,
            b_scale_base,
            permute_weights,
            permute_token_idx,
            seq_len,
            output,
            self.num_sm,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.use_tma
        )
        return

num_sm = torch.cuda.get_device_properties(0).multi_processor_count
gemm1_aot = gemm1Kernel(num_sm)
gemm2_aot = gemm2Kernel(num_sm)

#################################################################################
#################################################################################
#################################################################################
#################################################################################


kernel_src = """
#include <stdio.h>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <device_launch_parameters.h>

struct FusedGatingData
{
    // input
    void* routing_logits; // [seq_len, 256]
    void* routing_bias; // [256]
    float routing_scaling_factor;

    // output
    void* routing_idx; // [seq_len, 8]
    void* routing_weights; // [seq_len, 8]
};

static constexpr int NUM_EXPERTS = 256;
static constexpr int NUM_SELECTED_EXPERTS = 8;
static constexpr int NUM_EXPERT_GROUPS = 8;
static constexpr int NUM_SELECTED_GROUPS = 4;


// grid: [seq_len] 每个 block 负责一个 token
// block: [256] 每个 thread 负责一个 expert
__global__ void fusedGatingKernel(
    FusedGatingData data
) {
    __shared__ __nv_bfloat16 smem_bias[NUM_EXPERTS];
    __shared__ float smem_logits_with_sigmoid_bias[NUM_EXPERTS];
    __shared__ float smem_group_sums[NUM_EXPERT_GROUPS]; // 跨 warp 归约

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;


    // 1. sigmoid + bias
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x; // 某个 token 对应的某 expert
    smem_bias[threadIdx.x] = ((__nv_bfloat16*)data.routing_bias)[threadIdx.x];
    float logit = ((float*)data.routing_logits)[global_idx];
    logit = 1.0f / (1.0f + expf(-logit));
    logit += __bfloat162float(smem_bias[threadIdx.x]);
    smem_logits_with_sigmoid_bias[threadIdx.x] = logit;

    // 2. warp 内求 top-2
    float top2_m1 = logit;
    float top2_m2 = -FLT_MAX;
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_m1 = __shfl_xor_sync(0xffffffff, top2_m1, mask);
        float other_m2 = __shfl_xor_sync(0xffffffff, top2_m2, mask);


        if (other_m1 > top2_m1) {
            // top2_m1 = other_m1;
            top2_m2 = max(top2_m1, other_m2);
            top2_m1 = other_m1; // 注意更新顺序
        } else if (other_m1 > top2_m2) {
            top2_m2 = other_m1;
        }

    }
    
    // sum
    float top2_sum = top2_m1 + top2_m2;
    if (lane_id == 0) {
        smem_group_sums[warp_id] = top2_sum;
    }

    __syncthreads();

    int selected_groups_idx[NUM_SELECTED_GROUPS];
    
    int selected_group_expert_idx[NUM_SELECTED_GROUPS];
    float selected_group_expert_score[NUM_SELECTED_GROUPS];

    int top_expert_idx[NUM_SELECTED_EXPERTS];
    float top_expert_score[NUM_SELECTED_EXPERTS];

    if (warp_id == 0) {
        // 3. warp0 的 thread0 内直接求 top4
        if (lane_id == 0) {
            float selected_groups_sums[NUM_SELECTED_GROUPS];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                selected_groups_idx[i] = -1;
                selected_groups_sums[i] = -FLT_MAX;
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float cur_idx = i;
                float cur_sum = smem_group_sums[i];

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (cur_sum > selected_groups_sums[j]) {
                        for (int k = 3; k > j; k--) {
                            selected_groups_idx[k] = selected_groups_idx[k - 1];
                            selected_groups_sums[k] = selected_groups_sums[k - 1];
                        }
                        selected_groups_idx[j] = cur_idx;
                        selected_groups_sums[j] = cur_sum;
                        break;
                    }
                }
            }
        }
        // lane0 线程广播
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; i++) {
            selected_groups_idx[i] = __shfl_sync(0xffffffff, selected_groups_idx[i], 0);
        }
        
        // 1 个 warp 内每个线程对应 4 experts，共 128 个
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; i++) {  // bound of params.mNumLimitedGroups
            auto groupIdx= selected_groups_idx[i];
            selected_group_expert_idx[i] = groupIdx * NUM_EXPERTS / NUM_EXPERT_GROUPS + lane_id;
            selected_group_expert_score[i] = smem_logits_with_sigmoid_bias[selected_group_expert_idx[i]];
        }

        
        // 4. 128 选 top-8
        // 4.1. 局部 4 个元素降序
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; i++) {
            for (int j = i + 1; j < NUM_SELECTED_GROUPS; j++) {
                if (selected_group_expert_score[j] > selected_group_expert_score[i]) {
                    float tmp_score = selected_group_expert_score[i];
                    selected_group_expert_score[i] = selected_group_expert_score[j];
                    selected_group_expert_score[j] = tmp_score;

                    int tmp_idx = selected_group_expert_idx[i];
                    selected_group_expert_idx[i] = selected_group_expert_idx[j];
                    selected_group_expert_idx[j] = tmp_idx;
                }
            }
        }
        // 4.2. 初始化局部 top-8 数组，不足的部分补 -FLT_MAX
        float thread_scores[NUM_SELECTED_EXPERTS];
        int thread_indices[NUM_SELECTED_EXPERTS];
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_GROUPS; ++i) {
            thread_scores[i] = selected_group_expert_score[i];
            thread_indices[i] = selected_group_expert_idx[i];
        }
        #pragma unroll
        for (int i = NUM_SELECTED_GROUPS; i < NUM_SELECTED_EXPERTS; ++i) {
            thread_scores[i] = -FLT_MAX;
            thread_indices[i] = -1;
        }
        // 4.3. 归并
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_scores[NUM_SELECTED_EXPERTS];
            int other_indices[NUM_SELECTED_EXPERTS];
            
            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                other_scores[i] = __shfl_down_sync(0xffffffff, thread_scores[i], offset);
                other_indices[i] = __shfl_down_sync(0xffffffff, thread_indices[i], offset);
            }

            // 归并 top8
            float merged_scores[NUM_SELECTED_EXPERTS];
            int merged_indices[NUM_SELECTED_EXPERTS];
            int p1 = 0, p2 = 0;
            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                if (thread_scores[p1] >= other_scores[p2]) {
                    merged_scores[i] = thread_scores[p1];
                    merged_indices[i] = thread_indices[p1];
                    p1++;
                } else {
                    merged_scores[i] = other_scores[p2];
                    merged_indices[i] = other_indices[p2];
                    p2++;
                }
            }
            
            // 写回当前线程
            #pragma unroll
            for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
                thread_scores[i] = merged_scores[i];
                thread_indices[i] = merged_indices[i];
            }
        }
        // 4.4. 广播
        #pragma unroll
        for (int i = 0; i < NUM_SELECTED_EXPERTS; ++i) {
            top_expert_score[i] = __shfl_sync(0xffffffff, thread_scores[i], 0);
            top_expert_idx[i] = __shfl_sync(0xffffffff, thread_indices[i], 0);
        }

        // 5. 前 8 个 thread 计算最终 weight，写回
        if (lane_id < NUM_SELECTED_EXPERTS) {
            auto selected_expert = top_expert_idx[lane_id];
            auto selected_score = top_expert_score[lane_id] - __bfloat162float(smem_bias[selected_expert]);
            
            float score_sum = selected_score;
            score_sum += __shfl_xor_sync(0xff, score_sum, 1);
            score_sum += __shfl_xor_sync(0xff, score_sum, 2);
            score_sum += __shfl_xor_sync(0xff, score_sum, 4);

            auto final_score = selected_score * data.routing_scaling_factor / score_sum;

            int write_idx = blockIdx.x * NUM_SELECTED_EXPERTS + lane_id;
            ((int*)data.routing_idx)[write_idx] = selected_expert;
            ((float*)data.routing_weights)[write_idx] = final_score;
        }
    }

}


void launchFusedGatingKernel(
    void* routing_logits,
    void* routing_bias,
    float routing_scaling_factor,
    void* routing_idx,
    void* routing_weights,
    int seq_len
) {
    FusedGatingData data;
    data.routing_logits = routing_logits;
    data.routing_bias = routing_bias;
    data.routing_scaling_factor = routing_scaling_factor;
    data.routing_idx = routing_idx;
    data.routing_weights = routing_weights;

    int threads_per_block = NUM_EXPERTS; // 256
    int num_blocks = seq_len; // 每个 token 一个 block

    fusedGatingKernel<<<num_blocks, threads_per_block>>>(data);
}


__global__ void countExpertKernel(
    const int* __restrict__ routing_idx, // [seq_len, 8]
    int* __restrict__ expert_counts,    // [32]
    int seq_len,
    int local_expert_offset
) {
    __shared__ int smem_counts[32];
    
    int tid = threadIdx.x;
    if (tid < 32) {
        smem_counts[tid] = 0;
    }
    __syncthreads();

    // 每个 thread 处理 1 个 token
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int e_id = routing_idx[idx * 8 + k];
            if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                atomicAdd(&smem_counts[e_id - local_expert_offset], 1);
            }
        }
    }
    __syncthreads();

    // 3. 将 Block 结果汇总到 Global Memory
    if (tid < 32) {
        if (smem_counts[tid] > 0) {
            atomicAdd(&expert_counts[tid], smem_counts[tid]);
        }
    }
}

__global__ void exclusiveScan32Kernel(
    int* __restrict__ expert_counts, // [32]
    int* __restrict__ expert_offsets,      // [32]
    int* __restrict__ total_tokens         // [1]
) {
    __shared__ int temp[32];
    const int tid = threadIdx.x;

    if (tid < 32) {
        temp[tid] = expert_counts[tid];
    }
    __syncthreads();

    // In-place inclusive scan for fixed size 32.
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int add_val = 0;
        if (tid >= offset) {
            add_val = temp[tid - offset];
        }
        __syncthreads();
        if (tid < 32) {
            temp[tid] += add_val;
        }
        __syncthreads();
    }

    if (tid < 32) {
        expert_offsets[tid] = (tid == 0) ? 0 : temp[tid - 1];
        expert_counts[tid] = expert_offsets[tid];
    }
    if (tid == 31) {
        expert_offsets[32] = temp[31];
        total_tokens[0] = temp[31];
    }
}


__global__ void permuteKernel(
    const int* __restrict__ routing_idx,     // [seq_len, 8]
    const float* __restrict__ routing_weight,   // [seq_len, 8]
    int* __restrict__ expert_offsets,       // [32] - 传入前已做过 cumsum
    int* __restrict__ out_token_idx,        // [total_tokens]
    float* __restrict__ out_weights,        // [total_tokens]
    int* __restrict__ token2permuted_idx,     // [seq_len * 8]
    int* __restrict__ token_counts,
    int seq_len,
    int local_expert_offset
) {
    // 每个 thread 处理 1 个 token
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    int expert_cnt = 0;
    for (int k = 0; k < 8; ++k) {
        int e_id = routing_idx[idx * 8 + k];
        if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
            float w = routing_weight[idx * 8 + k];
            int rel_id = e_id - local_expert_offset;
            
            // 抢占全局位置
            int write_pos = atomicAdd(&expert_offsets[rel_id], 1);
            
            // 写入映射表
            out_token_idx[write_pos] = idx;
            out_weights[write_pos] = w;

            token2permuted_idx[idx * 8 + expert_cnt] = write_pos;
            expert_cnt++;
        }
    }
    token_counts[idx] = expert_cnt;
}

__global__ void countScanPermuteKernel(
    const int* __restrict__ routing_idx,      // [seq_len, 8]
    const float* __restrict__ routing_weight, // [seq_len, 8]
    int* __restrict__ expert_counts,          // [32] (kept as prefix offsets for compatibility)
    int* __restrict__ expert_offsets,         // [33]
    int* __restrict__ total_tokens,           // [1]
    int* __restrict__ out_token_idx,          // [total_tokens]
    float* __restrict__ out_weights,          // [total_tokens]
    int* __restrict__ token2permuted_idx,     // [seq_len * 8]
    int* __restrict__ token_counts,
    int seq_len,
    int local_expert_offset
) {
    __shared__ int smem_counts[32];
    __shared__ int smem_cursors[32];

    const int tid = threadIdx.x;
    if (tid < 32) {
        smem_counts[tid] = 0;
    }
    __syncthreads();

    // Pass-1: count local expert tokens.
    for (int idx = tid; idx < seq_len; idx += blockDim.x) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int e_id = routing_idx[idx * 8 + k];
            if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                atomicAdd(&smem_counts[e_id - local_expert_offset], 1);
            }
        }
    }
    __syncthreads();

    // Build exclusive offsets and initialize per-expert cursors.
    if (tid == 0) {
        int prefix = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            expert_offsets[i] = prefix;
            expert_counts[i] = prefix;
            smem_cursors[i] = prefix;
            prefix += smem_counts[i];
        }
        expert_offsets[32] = prefix;
        total_tokens[0] = prefix;
    }
    __syncthreads();

    // Pass-2: permute into compact buffers.
    for (int idx = tid; idx < seq_len; idx += blockDim.x) {
        int expert_cnt = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int e_id = routing_idx[idx * 8 + k];
            if (e_id >= local_expert_offset && e_id < local_expert_offset + 32) {
                const int rel_id = e_id - local_expert_offset;
                const int write_pos = atomicAdd(&smem_cursors[rel_id], 1);
                out_token_idx[write_pos] = idx;
                out_weights[write_pos] = routing_weight[idx * 8 + k];
                token2permuted_idx[idx * 8 + expert_cnt] = write_pos;
                expert_cnt++;
            }
        }
        token_counts[idx] = expert_cnt;
    }
}


__global__ void moe_permute_copy_fp8_with_scale_kernel(
    const __nv_fp8_e4m3* __restrict__ input,        // [S, 7168]
    const float* __restrict__ input_scale,          // [56, S]
    const int* __restrict__ permuted_token_idx,     // [TotalValidTokens]
    __nv_fp8_e4m3* __restrict__ output,             // [TotalValidTokens, 7168]
    float* __restrict__ output_scale,               // [56, TotalValidTokens]
    int* __restrict__ offset,                     // [33]
    int input_seq_len
) {
    const int HIDDEN_DIM = 7168;
    const int NUM_HIDDEN_BLOCKS = 56;
    const int VEC_SIZE = 16; // 128 bit / 8 bit = 16 elements per uint4

    int out_row_idx = blockIdx.x;
    if (out_row_idx >= offset[32]) return;

    int src_row_idx = permuted_token_idx[out_row_idx];

    const uint4* src_ptr4 = reinterpret_cast<const uint4*>(input + src_row_idx * HIDDEN_DIM);
    uint4* dst_ptr4 = reinterpret_cast<uint4*>(output + out_row_idx * HIDDEN_DIM);

    for (int v = threadIdx.x; v < HIDDEN_DIM / VEC_SIZE; v += blockDim.x) {
        dst_ptr4[v] = src_ptr4[v];
    }

    for (int hb = threadIdx.x; hb < NUM_HIDDEN_BLOCKS; hb += blockDim.x) {
        output_scale[hb * input_seq_len * 8 + out_row_idx] = input_scale[hb * input_seq_len + src_row_idx];
    }
}


void launchCountExpertAndOffsetsKernel(
    void* routing_idx,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    int seq_len,
    int local_expert_offset
) {
    cudaMemsetAsync(expert_counts, 0, 32 * sizeof(int));
    constexpr int threads_per_block = 256;
    const int num_blocks = (seq_len + threads_per_block - 1) / threads_per_block;
    countExpertKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<int*>(expert_counts),
        seq_len,
        local_expert_offset
    );

    exclusiveScan32Kernel<<<1, 32>>>(
        static_cast<int*>(expert_counts),
        static_cast<int*>(expert_offsets),
        static_cast<int*>(total_tokens)
    );
}

void launchPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_offsets,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
) {
    constexpr int threads_per_block = 256;
    const int num_blocks = (seq_len + threads_per_block - 1) / threads_per_block;
    permuteKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<const float*>(routing_weight),
        static_cast<int*>(expert_offsets),
        static_cast<int*>(out_token_idx),
        static_cast<float*>(out_weights),
        static_cast<int*>(token2permuted_idx),
        static_cast<int*>(token_counts),
        seq_len,
        local_expert_offset
    );
}

void launchCountScanPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
) {
    constexpr int threads_per_block = 256;
    countScanPermuteKernel<<<1, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<const float*>(routing_weight),
        static_cast<int*>(expert_counts),
        static_cast<int*>(expert_offsets),
        static_cast<int*>(total_tokens),
        static_cast<int*>(out_token_idx),
        static_cast<float*>(out_weights),
        static_cast<int*>(token2permuted_idx),
        static_cast<int*>(token_counts),
        seq_len,
        local_expert_offset
    );
}

void launchMoePermuteCopyFp8WithScaleKernel(
    void* input,
    void* input_scale,
    void* permuted_token_idx,
    void* output,
    void* output_scale,
    void* offset,
    int input_seq_len
) {
    constexpr int threads_per_block = 256;
    const int num_blocks = input_seq_len * 8;
    moe_permute_copy_fp8_with_scale_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const __nv_fp8_e4m3*>(input),
        static_cast<const float*>(input_scale),
        static_cast<const int*>(permuted_token_idx),
        static_cast<__nv_fp8_e4m3*>(output),
        static_cast<float*>(output_scale),
        static_cast<int*>(offset),
        input_seq_len
    );
}




__global__ void scatter_add_kernel(
    const float* __restrict__ tmp_output, // [seq*8, 7168]
    const int* __restrict__ token_idx,    // [seq*8]
    float* __restrict__ output,           // [seq, 7168]
    int* offset
) {
    int num_valid = offset[32];
    int row_idx = blockIdx.x;
    if (row_idx >= num_valid) return;

    int target_row = token_idx[row_idx];
    
    int col_offset = threadIdx.x * 4;

    for (; col_offset < 7168; col_offset += blockDim.x * 4) {
        float4 val = reinterpret_cast<const float4*>(&tmp_output[row_idx * 7168 + col_offset])[0];

        float* target_ptr = &output[target_row * 7168 + col_offset];
        
        atomicAdd(&target_ptr[0], val.x);
        atomicAdd(&target_ptr[1], val.y);
        atomicAdd(&target_ptr[2], val.z);
        atomicAdd(&target_ptr[3], val.w);
    }
}

void launchScatterAddKernel(
    void* tmp_output,
    void* token_idx,
    void* output,
    void* offset,
    int seq_len
) {
    size_t zero_size = seq_len * 7168 * sizeof(float);
    cudaMemsetAsync(output, 0, zero_size);

    constexpr int threads_per_block = 256;
    const int num_blocks = seq_len * 8;
    scatter_add_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float*>(tmp_output),
        static_cast<const int*>(token_idx),
        static_cast<float*>(output),
        static_cast<int*>(offset)
    );
}

__global__ void act_quant_kernel(
    const __half* __restrict__ input, // [valid, 4096], fp16
    __nv_fp8_e4m3* __restrict__ output,      // [valid, 2048]
    float* __restrict__ scale,       // [2048 / 128, valid]
    int* __restrict__ offset,        // [33]
    int seq_len
) {
    constexpr int HIDDEN = 2048;
    constexpr int GROUP = 128;
    constexpr float FP8_E4M3_MAX = 448.0f;

    const int row = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int num_valid = offset[32];
    if (row >= num_valid || group_idx >= (HIDDEN / GROUP) || tid >= GROUP) {
        return;
    }

    const int col = group_idx * GROUP + tid;
    const int in_row_base = row * (HIDDEN * 2);
    const int out_row_base = row * HIDDEN;

    // input 被拆成前后两半：input1 和 input2，然后做 input1 * SiLU(input2)。
    const float x1 = __half2float(input[in_row_base + col]);
    const float x2 = __half2float(input[in_row_base + HIDDEN + col]);
    const float silu2 = x2 / (1.0f + expf(-x2));
    const float res = x1 * silu2;

    __shared__ float smem_abs_max[GROUP];
    smem_abs_max[tid] = fabsf(res);
    __syncthreads();

    for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_abs_max[tid] = fmaxf(smem_abs_max[tid], smem_abs_max[tid + stride]);
        }
        __syncthreads();
    }

    const float max_abs = smem_abs_max[0];
    const float group_scale = (max_abs > 0.0f) ? (max_abs / FP8_E4M3_MAX) : 1.0f;
    if (tid == 0) {
        scale[group_idx * seq_len * 8 + row] = group_scale;
    }

    float q = res / group_scale;
    q = fminf(fmaxf(q, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    output[out_row_base + col] = static_cast<__nv_fp8_e4m3>(q);
}

void launchActQuantKernel(
    void* input,
    void* output,
    void* scale,
    void* offset,
    int seq_len
) {
    constexpr int threads_per_block = 128;
    const dim3 grid_dim(seq_len * 8, 2048 / 128);
    act_quant_kernel<<<grid_dim, threads_per_block>>>(
        static_cast<const __half*>(input),
        static_cast<__nv_fp8_e4m3*>(output),
        static_cast<float*>(scale),
        static_cast<int*>(offset),
        seq_len
    );
}

__global__ void reduce_add_kernel(
    const __nv_bfloat16* __restrict__ tmp_output, // [seq*8, 7168]
    const int* __restrict__ token2permuted_idx,     // [seq_len * 8]
    const int* __restrict__ token_counts,           // [seq_len]
    __nv_bfloat16* __restrict__ output           // [seq, 7168]
) {
    uint32_t token_id = blockIdx.x >> 3; // blockIdx.x / 8
    uint32_t inter_id = blockIdx.x & 7; // blockIdx.x % 8

    // Explicit 128-bit vectorized path: 1 x uint4 = 8 x bf16 = 4 x bf16x2.
    int col_bf16 = threadIdx.x * 8 + inter_id * 1024;
    // 每个 thread 一次处理 8 个 bf16，8 * 256 = 1024 个 bf16
    if (col_bf16 < 7168) {
        __nv_bfloat162 val2[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            val2[j] = __halves2bfloat162(__float2bfloat16(0.0f), __float2bfloat16(0.0f));
        }
        for (int i = 0; i < token_counts[token_id]; ++i) {
            int permuted_idx = token2permuted_idx[token_id * 8 + i];
            uint4 tmp_pack = reinterpret_cast<const uint4*>(
                &tmp_output[permuted_idx * 7168 + col_bf16]
            )[0];
            const __nv_bfloat162* tmp_vals2 = reinterpret_cast<const __nv_bfloat162*>(&tmp_pack);

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                val2[j] = __hadd2(val2[j], tmp_vals2[j]);
            }
        }

        uint4 out_pack;
        __nv_bfloat162* out_vals2 = reinterpret_cast<__nv_bfloat162*>(&out_pack);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out_vals2[j] = val2[j];
        }

        reinterpret_cast<uint4*>(&output[token_id * 7168 + col_bf16])[0] = out_pack;
    }
}

void launchReduceAddKernel(
    void* tmp_output,
    void* token2permuted_idx,
    void* token_counts,
    void* output,
    int seq_len
) {
    size_t zero_size = seq_len * 7168 * sizeof(__nv_bfloat16);
    cudaMemsetAsync(output, 0, zero_size);

    constexpr int threads_per_block = 256;
    // 让一个 block 处理一个 token 的 1024 个 bf16
    // 一个 token 对应于 7 个 block
    const int num_blocks = seq_len * 8;
    reduce_add_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const __nv_bfloat16*>(tmp_output),
        static_cast<const int*>(token2permuted_idx),
        static_cast<const int*>(token_counts),
        static_cast<__nv_bfloat16*>(output)
    );
}
"""

cpp_src = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")

void launchFusedGatingKernel(
    void* routing_logits,
    void* routing_bias,
    float routing_scaling_factor,
    void* routing_idx,
    void* routing_weights,
    int seq_len
);

void launchCountExpertAndOffsetsKernel(
    void* routing_idx,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    int seq_len,
    int local_expert_offset
);

void launchPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_offsets,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
);

void launchCountScanPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
);


void launchMoePermuteCopyFp8WithScaleKernel(
    void* input,
    void* input_scale,
    void* permuted_token_idx,
    void* output,
    void* output_scale,
    void* offset,
    int input_seq_len
);

void launchScatterAddKernel(
    void* tmp_output,
    void* token_idx,
    void* output,
    void* offset,
    int seq_len
);

void launchActQuantKernel(
    void* input,
    void* output,
    void* scale,
    void* offset,
    int seq_len
);

void fusedRoutePermuteCopyIntoWrapper(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    float routing_scaling_factor,
    torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale,
    int64_t local_expert_offset,
    torch::Tensor routing_idx,
    torch::Tensor routing_weights,
    torch::Tensor expert_counts,
    torch::Tensor expert_offsets,
    torch::Tensor total_tokens_device,
    torch::Tensor permute_token_idx,
    torch::Tensor permute_weight,
    torch::Tensor permute_hidden_states,
    torch::Tensor permute_hidden_states_scale,
    torch::Tensor token2permuted_idx,
    torch::Tensor token_counts,
    int seq_len
) {
    CHECK_CUDA(routing_logits);
    CHECK_CUDA(routing_bias);
    CHECK_CUDA(hidden_states);
    CHECK_CUDA(hidden_states_scale);
    CHECK_CUDA(routing_idx);
    CHECK_CUDA(routing_weights);
    CHECK_CUDA(expert_counts);
    CHECK_CUDA(expert_offsets);
    CHECK_CUDA(total_tokens_device);
    CHECK_CUDA(permute_token_idx);
    CHECK_CUDA(permute_weight);
    CHECK_CUDA(permute_hidden_states);
    CHECK_CUDA(permute_hidden_states_scale);

    CHECK_CONTIGUOUS(routing_logits);
    CHECK_CONTIGUOUS(routing_bias);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(hidden_states_scale);
    CHECK_CONTIGUOUS(routing_idx);
    CHECK_CONTIGUOUS(routing_weights);
    CHECK_CONTIGUOUS(expert_counts);
    CHECK_CONTIGUOUS(expert_offsets);
    CHECK_CONTIGUOUS(total_tokens_device);
    CHECK_CONTIGUOUS(permute_token_idx);
    CHECK_CONTIGUOUS(permute_weight);
    CHECK_CONTIGUOUS(permute_hidden_states);
    CHECK_CONTIGUOUS(permute_hidden_states_scale);

    CHECK_FLOAT32(routing_logits);
    CHECK_FLOAT32(hidden_states_scale);
    CHECK_INT32(routing_idx);
    CHECK_FLOAT32(routing_weights);
    CHECK_INT32(expert_counts);
    CHECK_INT32(expert_offsets);
    CHECK_INT32(total_tokens_device);
    CHECK_INT32(permute_token_idx);
    CHECK_FLOAT32(permute_weight);
    CHECK_FLOAT32(permute_hidden_states_scale);

    TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be [seq_len, 256]");
    TORCH_CHECK(routing_logits.size(1) == 256, "routing_logits second dim must be 256");
    TORCH_CHECK(routing_bias.numel() == 256, "routing_bias must have 256 elements");
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [seq_len, 7168]");
    TORCH_CHECK(hidden_states.size(1) == 7168, "hidden_states second dim must be 7168");
    TORCH_CHECK(hidden_states.element_size() == 1, "hidden_states must be 1-byte dtype for fp8 kernel");
    TORCH_CHECK(hidden_states_scale.dim() == 2, "hidden_states_scale must be [56, seq_len]");
    TORCH_CHECK(hidden_states_scale.size(0) == 56, "hidden_states_scale first dim must be 56");
    TORCH_CHECK(hidden_states_scale.size(1) == hidden_states.size(0), "hidden_states_scale second dim must equal hidden_states seq_len");
    TORCH_CHECK(routing_logits.size(0) == hidden_states.size(0), "routing_logits and hidden_states seq_len must match");


    launchFusedGatingKernel(
        routing_logits.data_ptr<float>(),
        routing_bias.data_ptr<at::BFloat16>(),
        routing_scaling_factor,
        routing_idx.data_ptr<int>(),
        routing_weights.data_ptr<float>(),
        static_cast<int>(seq_len)
    );

    if (seq_len > 256) {
        launchCountExpertAndOffsetsKernel(
            routing_idx.data_ptr<int>(),
            expert_counts.data_ptr<int>(),
            expert_offsets.data_ptr<int>(),
            total_tokens_device.data_ptr<int>(),
            static_cast<int>(seq_len),
            static_cast<int>(local_expert_offset)
        );
        launchPermuteKernel(
            routing_idx.data_ptr<int>(),
            routing_weights.data_ptr<float>(),
            expert_counts.data_ptr<int>(),
            permute_token_idx.data_ptr<int>(),
            permute_weight.data_ptr<float>(),
            token2permuted_idx.data_ptr<int>(),
            token_counts.data_ptr<int>(),
            static_cast<int>(seq_len),
            static_cast<int>(local_expert_offset)
        );
    } else {
        launchCountScanPermuteKernel(
        routing_idx.data_ptr<int>(),
        routing_weights.data_ptr<float>(),
        expert_counts.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        total_tokens_device.data_ptr<int>(),
        permute_token_idx.data_ptr<int>(),
        permute_weight.data_ptr<float>(),
        token2permuted_idx.data_ptr<int>(),
        token_counts.data_ptr<int>(),
        static_cast<int>(seq_len),
        static_cast<int>(local_expert_offset)
    );
    }

    launchMoePermuteCopyFp8WithScaleKernel(
        hidden_states.data_ptr(),
        hidden_states_scale.data_ptr<float>(),
        permute_token_idx.data_ptr<int>(),
        permute_hidden_states.data_ptr(),
        permute_hidden_states_scale.data_ptr<float>(),
        expert_offsets.data_ptr<int>(),
        static_cast<int>(seq_len)
    );
}

void scatterAddWrapper(
    torch::Tensor tmp_output,
    torch::Tensor token_idx,
    torch::Tensor output,
    torch::Tensor offset,
    int seq_len
) {
    CHECK_CUDA(tmp_output);
    CHECK_CUDA(token_idx);
    CHECK_CUDA(output);
    CHECK_CUDA(offset);
    CHECK_CONTIGUOUS(tmp_output);
    CHECK_CONTIGUOUS(token_idx);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(offset);
    CHECK_FLOAT32(tmp_output);
    CHECK_INT32(token_idx);
    CHECK_INT32(offset);

    TORCH_CHECK(tmp_output.dim() == 2, "tmp_output must be [TotalValidTokens, HiddenDim]");
    TORCH_CHECK(token_idx.dim() == 1, "token_idx must be [TotalValidTokens]");
    TORCH_CHECK(output.dim() == 2, "output must be [SeqLen, HiddenDim]");
    // TORCH_CHECK(offset.numel() == 1, "offset must have 1 element");
    TORCH_CHECK(tmp_output.size(0) == token_idx.size(0), "tmp_output and token_idx first dim must match");
    TORCH_CHECK(tmp_output.size(1) == output.size(1), "tmp_output and output second dim must match");

    launchScatterAddKernel(
        tmp_output.data_ptr<float>(),
        token_idx.data_ptr<int>(),
        output.data_ptr<float>(),
        offset.data_ptr<int>(),
        seq_len
    );
}

void actQuantWrapper(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor offset,
    int seq_len
) {
    launchActQuantKernel(
        input.data_ptr(),
        output.data_ptr(),
        scale.data_ptr(),
        offset.data_ptr(),
        seq_len
    );
}

void launchReduceAddKernel(
    void* tmp_output,
    void* token2permuted_idx,
    void* token_counts,
    void* output,
    int seq_len
);

void reduceAddWrapper(
    torch::Tensor tmp_output,
    torch::Tensor token2permuted_idx,
    torch::Tensor token_counts,
    torch::Tensor output,
    int seq_len
) {
    launchReduceAddKernel(
        tmp_output.data_ptr(),
        token2permuted_idx.data_ptr<int>(),
        token_counts.data_ptr<int>(),
        output.data_ptr<at::BFloat16>(),
        seq_len
    );
}
"""


my_lib = load_inline(
    name = "fused_gating",
    cuda_sources=kernel_src,
    cpp_sources=cpp_src,
    functions=["fusedRoutePermuteCopyIntoWrapper", "scatterAddWrapper", "actQuantWrapper", "reduceAddWrapper"],
    verbose=True,
    extra_cflags=['-O3', '-march=native'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

print("load succ")

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)


#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    • FP8 block-scale dequantization: float ≈ fp8 * scale
    • DeepSeek-V3 no-aux routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
        on the kept groups, take global top_k=8 experts
        combine with weights derived from s (without bias), normalized and
        scaled by routed_scaling_factor
    • Local computation:
        only experts in [local_expert_offset, local_expert_offset + E_local) are
        computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
        accumulation.
    """
    # print("\n*************** std start *****************")

    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    
    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168" 
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    # Block counts
    num_hidden_blocks = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, num_gemm1_out_blocks, num_hidden_blocks)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, num_hidden_blocks, num_intermediate_blocks)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # 1) FP8 block-scale dequantization
    # hidden_states: [T, H], scale: [H/128, T] (transposed layout)
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)                # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()            # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)                                   # [T, H/128, 128]
        .reshape(T, H)                                         # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                              # [T, H] float32

    # W13: [E_local, 2I, H], scale: [E_local, (2I)/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)  # [E, 2I, H/128]
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  # [E, 2I, H]
    W13 = W13_fp32 * S13_expanded                              # [E, 2I, H] float32

    # W2: [E_local, H, I], scale: [E_local, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)    # [E, H, I/128]
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)    # [E, H, I]
    W2 = W2_fp32 * S2_expanded                                 # [E, H, I] float32

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)                      # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)              # [E_global]

    # Sigmoid
    s = 1.0 / (1.0 + torch.exp(-logits))                       # [T, E]
    s_with_bias = s + bias                                     # [T, E] (broadcast)

    # Grouping
    group_size = E_global // N_GROUP # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)    # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)                        # [T, 8]

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  # [T, 4]
    group_mask = torch.zeros_like(group_scores)                # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)  # [T, E]

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)                  # [T, E]
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)  # [T, 8]

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)                                    # [T, E]
    M.scatter_(1, topk_idx, 1.0)                               # 0/1 mask
    weights = s * M                                            # [T, E]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor  # [T, E]

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1→SwiGLU→GEMM2, accumulate by weights
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Tokens that selected this global expert ge in their top-k
        sel_mask_per_token = (topk_idx == ge).any(dim=1)       # [T] bool
        if not sel_mask_per_token.any():
            continue
        # print(sel_mask_per_token)
        # print(f"expert idx: {le}, tokens: {sel_mask_per_token.sum()}")
        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  # [Tk]
        Tk = token_idx.numel()

        # Gather inputs and weights for this expert
        A_e = A.index_select(0, token_idx)                     # [Tk, H]
        W13_e = W13[le]                                        # [2I, H]
        W2_e = W2[le]                                          # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())                             # [Tk, 2I]

        # SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
        X1 = G1[:, :I]                                         # [Tk, I]
        X2 = G1[:, I:]                                         # [Tk, I]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))                  # [Tk, I]
        C = silu_X2 * X1                                       # [Tk, I]

        # if le == 8:
        #     print(C[0][:10])
        #     print(f"std gemm1 shape: {C.shape}")
        #     print(f"std x1: {X1[0][:10]}")
        #     print(f"std x2: {X2[0][:10]}")
        #     return C

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())                                 # [Tk, H]

        # Accumulate with per-token routing weights for this expert
        w_tok = weights.index_select(0, token_idx)[:, ge]      # [Tk]
        # if le == 8:
        #     print(O)
        #     print(w_tok)
        # print(w_tok)
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))  # [Tk,H] * [Tk,1]

    # print("*************** std over *****************\n")

    return output.to(torch.bfloat16)

def test_time(impl_func, *args):
    # warmup
    for _ in range(20):
        impl_func(*args)
    torch.cuda.synchronize()

    n_iters = 100
    start_time = time.time()
    for _ in range(n_iters):
        impl_func(*args)
        torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) * 1000 / n_iters
    print(f"Average execution time over {n_iters} runs: {avg_time:.6f} ms")



class FusedMoeWorkspace:
    def __init__(self, seq_len: int, device: torch.device):
        total_tokens = seq_len * 8
        self.seq_len = seq_len
        self.total_tokens = total_tokens

        # fused_route_permute_copy_into outputs / scratch
        self.routing_idx = torch.empty((seq_len, 8), device=device, dtype=torch.int32)
        self.routing_weights = torch.empty((seq_len, 8), device=device, dtype=torch.float32)
        self.expert_counts = torch.empty((32,), device=device, dtype=torch.int32)
        self.expert_offsets = torch.empty((33,), device=device, dtype=torch.int32)
        self.total_tokens_device = torch.empty((1,), device=device, dtype=torch.int32)
        self.permute_token_idx = torch.empty((total_tokens,), device=device, dtype=torch.int32)
        self.permute_weight = torch.empty((total_tokens,), device=device, dtype=torch.float32)
        self.permute_hidden_states = torch.empty((total_tokens, 7168), device=device, dtype=torch.float8_e4m3fn)
        self.permute_hidden_states_scale = torch.empty((56, total_tokens), device=device, dtype=torch.float32)
        self.token2permuted_idx = torch.empty((seq_len * 8,), device=device, dtype=torch.int32)
        self.token_counts = torch.empty((seq_len,), device=device, dtype=torch.int32)

        # gemm / reduce outputs
        self.gemm1_output = torch.empty((total_tokens, 2048), device=device, dtype=torch.float16)
        self.gemm2_input = torch.empty((total_tokens, 2048), device=device, dtype=torch.float8_e4m3fn)
        self.gemm2_input_scale = torch.empty((2048 // 128, total_tokens), device=device, dtype=torch.float32)
        self.gemm2_output = torch.empty((total_tokens, 7168), device=device, dtype=torch.bfloat16)
        self.output = torch.empty((seq_len, 7168), device=device, dtype=torch.bfloat16)

        self.stream = torch.cuda.current_stream()


_MAX_SEQ_LEN = 15000
_FUSED_MOE_WORKSPACE = None


def _get_fused_moe_workspace(device: torch.device) -> FusedMoeWorkspace:
    global _FUSED_MOE_WORKSPACE
    if _FUSED_MOE_WORKSPACE is None:
        _FUSED_MOE_WORKSPACE = FusedMoeWorkspace(_MAX_SEQ_LEN, device)
    return _FUSED_MOE_WORKSPACE


@torch.no_grad()
def fused_moe(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    seq_len = hidden_states.size(0)
    ws = _get_fused_moe_workspace(hidden_states.device)
    stream = ws.stream
    
    # print("\n*************** fused start *****************")
    # 1~3. routing + fused permute + copy
    my_lib.fusedRoutePermuteCopyIntoWrapper(
        routing_logits,
        routing_bias,
        routed_scaling_factor,
        hidden_states,
        hidden_states_scale,
        local_expert_offset,
        ws.routing_idx,
        ws.routing_weights,
        ws.expert_counts,
        ws.expert_offsets,
        ws.total_tokens_device,
        ws.permute_token_idx,
        ws.permute_weight,
        ws.permute_hidden_states,
        ws.permute_hidden_states_scale,
        ws.token2permuted_idx,
        ws.token_counts,
        seq_len
    )

    # 4. gemm1 & activation
    # gemm1(
    #     permute_hidden_states,
    #     permute_hidden_states_scale,
    #     offset,
    #     seq_len,
    #     gemm1_weights,
    #     gemm1_weights_scale,
    #     gemm1_output,
    #     gemm1_output_scale,
    #     num_sm=num_sm,
    # )
    gemm1_aot(
        ws.permute_hidden_states,
        ws.permute_hidden_states_scale,
        ws.expert_offsets,
        gemm1_weights,
        gemm1_weights_scale,
        seq_len,
        ws.gemm1_output,
        ws.gemm2_input_scale,
        stream,
    )
    my_lib.actQuantWrapper(
        ws.gemm1_output,
        ws.gemm2_input,
        ws.gemm2_input_scale,
        ws.expert_offsets,
        seq_len,
    )
    
    # 5. gemm2 & output
    # gemm2(
    #     gemm1_output,
    #     gemm1_output_scale,
    #     offset,
    #     gemm2_weights,
    #     gemm2_weights_scale,
    #     permute_weight,
    #     permute_token_idx,
    #     seq_len,
    #     output=gemm2_output,
    #     num_sm=num_sm,
    # )
    gemm2_aot(
        ws.gemm2_input,
        ws.gemm2_input_scale,
        ws.expert_offsets,
        gemm2_weights,
        gemm2_weights_scale,
        ws.permute_weight,
        ws.permute_token_idx,
        seq_len,
        ws.gemm2_output,
        stream,
    )

    my_lib.reduceAddWrapper(
        ws.gemm2_output,
        ws.token2permuted_idx,
        ws.token_counts,
        ws.output,
        seq_len,
    )

    # print("*************** fused over *****************\n")
    return ws.output[:seq_len]

def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, cfg
):
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    eps = 1e-8
    abs_error = torch.abs(x - y)
    rel_error = abs_error / (torch.abs(y) + eps)

    total_elements = abs_error.numel()
    if total_elements == 0:
        return 0.0, 0.0, False, 1.0

    required_matched_ratio = (
        cfg["required_matched_ratio"] if cfg["required_matched_ratio"] is not None else 0.9
    )
    exceeds_tol_mask = (abs_error > cfg["atol"]) & (rel_error > cfg["rtol"])
    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < required_matched_ratio

    max_abs = float(abs_error.max().item())
    max_rel = float(rel_error.max().item())

    return max_abs, max_rel, exceeds_tol, matched_ratio

import json
WORKLOAD_PATH = "/root/mlsys26-contest"
def read_workload(
    file_path=f"{WORKLOAD_PATH}/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl",
    idx=0,
):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    workload = data[idx]["workload"]
    seq_len = workload["axes"]["seq_len"]
    routing_logits_file = WORKLOAD_PATH + workload["inputs"]["routing_logits"]["path"][1:]
    routing_bias_file = WORKLOAD_PATH + workload["inputs"]["routing_bias"]["path"][1:]
    local_expert_offset = workload["inputs"]["local_expert_offset"]["value"]

    def read_safetensors(file_path, key):
        from safetensors import safe_open
        with safe_open(file_path, framework="pt", device='cuda') as f:
            tensor = f.get_tensor(key)
        return tensor
    
    routing_logits = read_safetensors(routing_logits_file, "routing_logits")
    routing_bias = read_safetensors(routing_bias_file, "routing_bias")
    print(f"test seq_len: {seq_len}")

    
    return seq_len, routing_logits, routing_bias, local_expert_offset

# workload_idx = 17
# seq_len, routing_logits, routing_bias, local_expert_offset = read_workload(idx=workload_idx)

# print("========== test ==========")
# torch.manual_seed(42)
# torch.set_default_device('cuda')
# torch.cuda.set_device(0)

# # seq_len = 32  # 示例序列长度
# num_experts = 256
# num_local_experts = 32  # 假设当前 Rank 负责的专家数
# hidden_size = 7168
# intermediate_size = 2048
# gemm1_out_size = 4096

# block_size = 128
# num_hidden_blocks = hidden_size // block_size
# num_intermediate_blocks = intermediate_size // block_size
# num_gemm1_out_blocks = gemm1_out_size // block_size


# # routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32)
# # routing_bias = torch.randn(num_experts, dtype=torch.bfloat16)
# hidden_states = torch.randn(seq_len, hidden_size).to(torch.float8_e4m3fn)
# hidden_states_scale = torch.randn(num_hidden_blocks, seq_len, dtype=torch.float32)
# gemm1_weights = torch.randn(
#     num_local_experts, gemm1_out_size, hidden_size
# ).to(torch.float8_e4m3fn)
# gemm1_weights_scale = torch.randn(
#     num_local_experts, num_gemm1_out_blocks, num_hidden_blocks, dtype=torch.float32
# )
# gemm2_weights = torch.randn(
#     num_local_experts, hidden_size, intermediate_size
# ).to(torch.float8_e4m3fn)
# gemm2_weights_scale = torch.randn(
#     num_local_experts, num_hidden_blocks, num_intermediate_blocks, dtype=torch.float32
# )
# # local_expert_offset = 32
# routed_scaling_factor = 1.11


# # with profile(
# #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
# #     with_stack=True,
# #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
# # ) as prof:
# test_time(fused_moe, routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor)
# # test_time(run, routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor)

# time.sleep(0.5)

# fused_output = fused_moe(
#     routing_logits,
#     routing_bias,
#     hidden_states,
#     hidden_states_scale,
#     gemm1_weights,
#     gemm1_weights_scale,
#     gemm2_weights,
#     gemm2_weights_scale,
#     local_expert_offset,
#     routed_scaling_factor
# )

# std_output = run(
#     routing_logits,
#     routing_bias,
#     hidden_states,
#     hidden_states_scale,
#     gemm1_weights,
#     gemm1_weights_scale,
#     gemm2_weights,
#     gemm2_weights_scale,
#     local_expert_offset,
#     routed_scaling_factor
# )


# max_abs, max_rel, exceeds_tol, matched_ratio = compute_error_stats(
#     fused_output,
#     std_output,
#     {
#         "atol": 1.0,
#         "rtol": 0.3,
#         "required_matched_ratio": 0.9,
#     }
# )
# print(f"Max absolute error: {max_abs:.6f}")
# print(f"Max relative error: {max_rel:.6f}")
# print(f"Matched ratio: {matched_ratio:.4%}")
# print(f"Exceeds tolerance: {exceeds_tol}")