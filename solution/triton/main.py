import torch
from torch.utils.cpp_extension import load, load_inline
import time

import torch.utils.cpp_extension

import triton
import triton.language as tl
from typing import List, Sequence



#################################################################################
#################################################################################
#################################################################################
#################################################################################

@triton.jit
def gemm1_kernel(
    # input
    group_a_ptrs, # [group_size], fp8 -> [s_i, 7168]
    a_scale_ptrs, # [group_size], fp32 -> [7168 // 128, s_i]
    group_b_ptrs, # [group_size], fp8 -> [4096, 7168]
    b_scale_ptrs, # [group_size], fp32 -> [4096 // 128, 7168 // 128]
    group_gemm_sizes, # [group_size, 3], <m, n, k>, n=4096, k=7168
    g_lds, # [group_size, 3], <lda, ldb, ldc>
    group_size, # num of gemms = num of local experts, <= 32
    # output
    group_c_ptrs, # [group_size], fp8 -> [s_i, 2048]
    # c_scale_ptrs, # [group_size], fp32 -> [2048 // 128, s_i]
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    dtype = tl.float8e4nv
    # dtype = tl.float32
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    # 遍历 gemm
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1) # 4096
        gk = tl.load(group_gemm_sizes + g * 3 + 2) # 7168
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn // 2, BLOCK_SIZE_N) # 处理两个 tile
        num_tiles = num_m_tiles * num_n_tiles

        # 检查当前 tile 编号是否还在当前 gemm 范围内
        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float32))

            a_scale_ptr = tl.load(a_scale_ptrs + g).to(tl.pointer_type(tl.float32))
            b_scale_ptr = tl.load(b_scale_ptrs + g).to(tl.pointer_type(tl.float32))
            # c_scale_ptr = tl.load(c_scale_ptrs + g).to(tl.pointer_type(tl.float32))

            # TMA
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
            b2_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
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
                offs_bn2 = (tile_n_idx + num_n_tiles) * BLOCK_SIZE_N

                # 缩放矩阵每一行的元素个数
                num_k_blocks = tl.cdiv(k, 128)

                acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load(
                        [offs_am, kk * BLOCK_SIZE_K],
                    )
                    b1 = b1_desc.load(
                        [offs_bn1, kk * BLOCK_SIZE_K],
                    )
                    b2 = b2_desc.load(
                        [offs_bn2, kk * BLOCK_SIZE_K],
                    )
                    
                    # 获取当前 128x128 block 的缩放因子
                    # row_idx * cols + col_idx
                    # a_scale = tl.load(a_scale_ptr + (offs_am // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                    a_scale_idx = a_scale_ptr + offs_am + kk * gm + tl.arange(0, BLOCK_SIZE_M)
                    a_scale_vec = tl.load(a_scale_idx, mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm, other=1.0)
                    a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1)) # 广播到 [M, 1]
                    b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128)) # 1 个数
                    b2_scale = tl.load(b_scale_ptr + (offs_bn2 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128)) # 1 个数


                    acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    acc2 += tl.dot(a, b2.T) * a_scale_col * b2_scale
                    # a = a.to(tl.float32) * a_scale_col
                    # b1 = b1.to(tl.float32) * b1_scale
                    # b2 = b2.to(tl.float32) * b2_scale
                    # acc1 += tl.dot(a, b1.T, input_precision="ieee")
                    # acc2 += tl.dot(a, b2.T, input_precision="ieee")
                    

                # activation
                silu_x2 = acc2 / (1.0 + tl.exp(-acc2))
                res_c = silu_x2 * acc1

                # if tile_idx == 0:
                #     tl.device_print("Partial Block:", res_c, acc1, acc2)

                # 量化
                # abs_c = tl.abs(res_c)
                # # 在 N 维度上做 reduce，找最大值
                # # max_val 的 shape 为 [BLOCK_SIZE_M]
                # max_val = tl.max(abs_c, axis=1)
                # fp8_max = 448.0
                # tile_scale = max_val / fp8_max
                # res_c = (res_c / tl.reshape(tile_scale, (BLOCK_SIZE_M, 1))).to(tl.float8e4nv)

                # 写回
                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N
                c_desc.store(
                    [offs_cm, offs_cn],
                    res_c,
                )
                # c_scale_idx = c_scale_ptr + offs_cm + tile_n_idx * gm + tl.arange(0, BLOCK_SIZE_M)
                # tl.store(c_scale_idx, tile_scale, mask=(offs_cm + tl.arange(0, BLOCK_SIZE_M)) < gm)


                # 下一个 tile
                tile_idx += NUM_SM

        # 进入下一个 gemm
        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles

def _to_ptr_tensor(tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    ptrs = [t.data_ptr() for t in tensors]
    return torch.tensor(ptrs, device="cpu", dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)

def launch_gemm1_kernel(
    a_list: Sequence[torch.Tensor],
    a_scale_list: Sequence[torch.Tensor],
    b_list: Sequence[torch.Tensor],
    b_scale_list: Sequence[torch.Tensor],
    out_list: Sequence[torch.Tensor],
    # out_scale_list: Sequence[torch.Tensor],
    *,
    block_size_m: int = 32,
    block_size_n: int = 128,
    block_size_k: int = 128,
) -> List[torch.Tensor]:
    """Launch grouped_matmul_tma_kernel with torch tensors.

    Args:
        a_list: list of A_i with shape [M_i, K_i], fp8.
        a_scale_list: list of per-block scales for A_i, float32.
        b_list: list of B_i with shape [N_i, K_i], fp8.
        b_scale_list: list of per-block scales for B_i, float32.

    Returns:
        List of C_i tensors, each with shape [M_i, N_i], bfloat16.
    """
    # _validate_group_inputs(a_list, a_scale_list, b_list, b_scale_list)
    # if block_size_k % 128 != 0:
    #     raise ValueError("block_size_k must be a multiple of 128 for current scale indexing")

    group_size = len(a_list)
    device = a_list[0].device

    # Keep row-major contiguous layout to match [stride, 1] descriptors.
    a_list = [a.contiguous() for a in a_list]
    b_list = [b.contiguous() for b in b_list]

    group_gemm_sizes = torch.empty((group_size, 3), device="cpu", dtype=torch.int32)
    g_lds = torch.empty((group_size, 3), device="cpu", dtype=torch.int32)

    for i, (a, b, c) in enumerate(zip(a_list, b_list, out_list)):
        m, k = a.shape
        n = b.shape[0]
        group_gemm_sizes[i, 0] = m
        group_gemm_sizes[i, 1] = n
        group_gemm_sizes[i, 2] = k
        g_lds[i, 0] = a.stride(0)
        g_lds[i, 1] = b.stride(0)
        g_lds[i, 2] = c.stride(0)

    group_gemm_sizes = group_gemm_sizes.to(device)
    g_lds = g_lds.to(device)

    group_a_ptrs = _to_ptr_tensor(a_list, device)
    group_b_ptrs = _to_ptr_tensor(b_list, device)
    group_c_ptrs = _to_ptr_tensor(out_list, device)
    a_scale_ptrs = _to_ptr_tensor(a_scale_list, device)
    b_scale_ptrs = _to_ptr_tensor(b_scale_list, device)
    # c_scale_ptrs = _to_ptr_tensor(out_scale_list, device)
    torch.cuda.synchronize(device)

    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (num_sm,)

    gemm1_kernel[grid](
        # input
        group_a_ptrs,
        a_scale_ptrs,
        group_b_ptrs,
        b_scale_ptrs,
        group_gemm_sizes,
        g_lds,
        group_size,
        # output
        group_c_ptrs,
        # c_scale_ptrs,
        NUM_SM=num_sm,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        # num_stages=1,
    )
    return

# input:
# [sum(s), 7168]
# -> 共 32 个 expert，sum(s) 可以分成 32 部分，每个部分的长度，以及每个部分的 seq id，即每个 expert 实际上是处理的哪些行
def gemm1(
    permute_hidden_states_list: torch.Tensor, # [sum(s), 7168], fp8
    hidden_states_scale_list: torch.Tensor, # [7168//128, seq_len]
    gemm1_weights_list: torch.Tensor, # [32, 4096, 7168], fp8
    gemm1_weights_scale_list: torch.Tensor, # [32, 4096//128, 7168//128], fp32
): # return [sum(s), 4096]
    
    out_list = []
    out_scale_list = []

    for i in range(len(permute_hidden_states_list)):
        s_i = permute_hidden_states_list[i].shape[0]
        out_list.append(torch.empty((s_i, 2048), device=permute_hidden_states_list[i].device, dtype=torch.float32))
        # out_scale_list.append(torch.empty((2048 // 128, s_i), device=permute_hidden_states_list[i].device, dtype=torch.float32))

    launch_gemm1_kernel(
        permute_hidden_states_list,
        hidden_states_scale_list,
        gemm1_weights_list,
        gemm1_weights_scale_list,
        out_list,
        # out_scale_list,
        block_size_m=32,
        block_size_n=128,
        block_size_k=128,
    )

    return out_list, out_scale_list

#################################################################################
#################################################################################
#################################################################################
#################################################################################

# local experts: 32
# 每个 expert: [s_i, h]
# input: 
# [sum(s_i), h]
# [sum(s_i), 1] -> weight
# output:
@triton.jit
def gemm2_kernel(
    # input
    group_a_ptrs, # [group_size], fp8 -> [s_i, 2048]
    # a_scale_ptrs, # [group_size], fp32 -> [2048 // 128, s_i]
    group_b_ptrs, # [group_size], fp8 -> [7168, 2048]
    b_scale_ptrs, # [group_size], fp32 -> [7168 // 128, 2048 // 128]
    group_gemm_sizes, # [group_size, 3], <m, n, k> n=7168,k=2048
    g_lds, # [group_size, 3], <lda, ldb, ldc>
    group_size, # num of gemms = num of local experts, 32
    permute_weights_ptrs, # [group_size], fp32 -> [s_i]
    permute_token_idx_ptrs, # [group_size], int32 -> [s_i]
    #output
    output_ptr, # [seq_len, 7168], bf16
    # other
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_gemm_end_tile_idx = 0

    # 遍历 gemm
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1) # 4096
        gk = tl.load(group_gemm_sizes + g * 3 + 2) # 7168
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # 检查当前 tile 编号是否还在当前 gemm 范围内
        if tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float32))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float8e4nv))
            
            # a_scale_ptr = tl.load(a_scale_ptrs + g).to(tl.pointer_type(tl.float32))
            b_scale_ptr = tl.load(b_scale_ptrs + g).to(tl.pointer_type(tl.float32))

            # 当前 expert 处理的 tokens 对应的 weight，长度为 gm
            p_weight_ptr = tl.load(permute_weights_ptrs + g).to(tl.pointer_type(tl.float32))
            # 当前 expert 处理的 tokens 对应的 token idx，长度为 gm，决定写回 output 的位置
            p_source_idx_ptr = tl.load(permute_token_idx_ptrs + g).to(tl.pointer_type(tl.int32))

            # TMA
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

            # 在当前 gemm 内拿多个 tile
            while (tile_idx >= last_gemm_end_tile_idx and tile_idx < last_gemm_end_tile_idx + num_tiles):
                k = gk
                tile_idx_in_gemm = tile_idx - last_gemm_end_tile_idx
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn1 = tile_n_idx * BLOCK_SIZE_N

                num_k_blocks = tl.cdiv(k, 128)
                acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load(
                        [offs_am, kk * BLOCK_SIZE_K],
                    )
                    b1 = b1_desc.load(
                        [offs_bn1, kk * BLOCK_SIZE_K],
                    )
                    
                    # 获取当前 128x128 block 的缩放因子
                    # row_idx * cols + col_idx
                    # a_scale = tl.load(a_scale_ptr + (offs_am // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128))
                    # a_scale_idx = a_scale_ptr + offs_am + kk * gm + tl.arange(0, BLOCK_SIZE_M)
                    # a_scale_vec = tl.load(a_scale_idx, mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < gm, other=1.0)
                    # a_scale_col = tl.reshape(a_scale_vec, (BLOCK_SIZE_M, 1)) # 广播到 [M, 1]
                    b1_scale = tl.load(b_scale_ptr + (offs_bn1 // 128) * num_k_blocks + (kk * BLOCK_SIZE_K // 128)) # 1 个数

                    # # acc1 += tl.dot(a, b1.T) * a_scale_col * b1_scale
                    # a = a.to(tl.float32) * a_scale_col
                    b1 = b1.to(tl.float32) * b1_scale
                    # acc1 += tl.dot(a, b1.T, input_precision="ieee")
                    acc1 += tl.dot(a, b1.T)

                # 和 weight 相乘
                offs_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offs_m < gm
                tile_weights = tl.load(p_weight_ptr + offs_m, mask=mask_m, other=0.0)
                acc1 = acc1 * tl.reshape(tile_weights, (BLOCK_SIZE_M, 1))

                # 写回 output: scatter-add
                source_indices = tl.load(p_source_idx_ptr + offs_m, mask=mask_m, other=0)
                offs_n = offs_bn1 + tl.arange(0, BLOCK_SIZE_N)
                mask_n = offs_n < gn
                source_indices_col = tl.reshape(source_indices, (BLOCK_SIZE_M, 1))
                offs_n_row = tl.reshape(offs_n, (1, BLOCK_SIZE_N))
                mask_m_col = tl.reshape(mask_m, (BLOCK_SIZE_M, 1))
                mask_n_row = tl.reshape(mask_n, (1, BLOCK_SIZE_N))
                dest_offsets = source_indices_col * 7168 + offs_n_row
                full_mask = mask_m_col & mask_n_row
                tl.atomic_add(output_ptr + dest_offsets, acc1, mask=full_mask)

                # 下一个 tile
                tile_idx += NUM_SM

        # 进入下一个 gemm
        last_gemm_end_tile_idx = last_gemm_end_tile_idx + num_tiles

def launch_gemm2_kernel(
    a_list: Sequence[torch.Tensor],
    # a_scale_list: Sequence[torch.Tensor],
    b_list: Sequence[torch.Tensor],
    b_scale_list: Sequence[torch.Tensor],
    permute_weights_list: Sequence[torch.Tensor],
    permute_token_idx_list: Sequence[torch.Tensor],
    output: torch.Tensor,
    *,
    block_size_m: int = 32,
    block_size_n: int = 128,
    block_size_k: int = 128,
) -> torch.Tensor:
    # _validate_gemm2_inputs(
    #     a_list,
    #     a_scale_list,
    #     b_list,
    #     b_scale_list,
    #     permute_weights_list,
    #     permute_token_idx_list,
    #     output,
    # )
    if block_size_k % 128 != 0:
        raise ValueError("block_size_k must be a multiple of 128 for current scale indexing")

    group_size = len(a_list)
    device = a_list[0].device

    # Keep row-major contiguous layout to match [stride, 1] descriptors.
    a_list = [a.contiguous() for a in a_list]
    b_list = [b.contiguous() for b in b_list]
    # a_scale_list = [a_s.contiguous() for a_s in a_scale_list]
    b_scale_list = [b_s.contiguous() for b_s in b_scale_list]
    permute_weights_list = [w.contiguous() for w in permute_weights_list]
    permute_token_idx_list = [idx.contiguous() for idx in permute_token_idx_list]
    output = output.contiguous()

    group_gemm_sizes = torch.empty((group_size, 3), device="cpu", dtype=torch.int32)
    g_lds = torch.empty((group_size, 3), device="cpu", dtype=torch.int32)

    for i, (a, b) in enumerate(zip(a_list, b_list)):
        m, k = a.shape
        n = b.shape[0]
        group_gemm_sizes[i, 0] = m
        group_gemm_sizes[i, 1] = n
        group_gemm_sizes[i, 2] = k
        g_lds[i, 0] = a.stride(0)
        g_lds[i, 1] = b.stride(0)
        g_lds[i, 2] = output.stride(0)

    group_gemm_sizes = group_gemm_sizes.to(device)
    g_lds = g_lds.to(device)

    group_a_ptrs = _to_ptr_tensor(a_list, device)
    # a_scale_ptrs = _to_ptr_tensor(a_scale_list, device)
    group_b_ptrs = _to_ptr_tensor(b_list, device)
    b_scale_ptrs = _to_ptr_tensor(b_scale_list, device)
    permute_weights_ptrs = _to_ptr_tensor(permute_weights_list, device)
    permute_token_idx_ptrs = _to_ptr_tensor(permute_token_idx_list, device)
    torch.cuda.synchronize(device)

    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (num_sm,)

    gemm2_kernel[grid](
        group_a_ptrs,
        # a_scale_ptrs,
        group_b_ptrs,
        b_scale_ptrs,
        group_gemm_sizes,
        g_lds,
        group_size,
        permute_weights_ptrs,
        permute_token_idx_ptrs,
        output,
        NUM_SM=num_sm,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        # num_stages=1,
    )
    return output

def gemm2(
    gemm1_output_fp8_list: List[torch.Tensor],
    # gemm1_output_scale_list: List[torch.Tensor],
    gemm2_weights_list: List[torch.Tensor],
    gemm2_weights_scale_list: List[torch.Tensor],
    permute_weights_list: List[torch.Tensor],
    permute_token_idx_list: List[torch.Tensor],
    seq_len: int
):
        
    output = torch.zeros((seq_len, 7168), device=gemm1_output_fp8_list[0].device, dtype=torch.float32)

    launch_gemm2_kernel(
        a_list=gemm1_output_fp8_list,
        # a_scale_list=gemm1_output_scale_list,
        b_list=gemm2_weights_list,
        b_scale_list=gemm2_weights_scale_list,
        permute_weights_list=permute_weights_list,
        permute_token_idx_list=permute_token_idx_list,
        output=output,
        block_size_m=32,
        block_size_n=128,
        block_size_k=128,
    )

    return output.to(torch.bfloat16)


#################################################################################
#################################################################################
#################################################################################
#################################################################################


fused_gating_src = """
#include <stdio.h>
#include <cfloat>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")

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
"""

fused_gating_cpp_src = """
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


std::tuple<torch::Tensor, torch::Tensor> fusedGatingWrapper(
    torch::Tensor routing_logits, 
    torch::Tensor routing_bias, 
    float routing_scaling_factor
) {
    auto routing_idx = torch::empty(
        {routing_logits.size(0), 8},
        torch::dtype(torch::kInt32).device(routing_logits.device())
    );
    auto routing_weights = torch::empty(
        {routing_logits.size(0), 8},
        torch::dtype(torch::kFloat32).device(routing_logits.device())
    );

    
    launchFusedGatingKernel(
        routing_logits.data_ptr<float>(),
        routing_bias.data_ptr<at::BFloat16>(),
        routing_scaling_factor,
        routing_idx.data_ptr<int>(),
        routing_weights.data_ptr<float>(),
        routing_logits.size(0)
    );

    return std::make_tuple(routing_idx, routing_weights);
}
"""

permute_src = """
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <device_launch_parameters.h>

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
    const int* __restrict__ expert_counts, // [32]
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
    }
    if (tid == 31) {
        total_tokens[0] = temp[31];
    }
}


__global__ void permuteKernel(
    const int* __restrict__ routing_idx,     // [seq_len, 8]
    const float* __restrict__ routing_weight,   // [seq_len, 8]
    int* __restrict__ expert_offsets,       // [32] - 传入前已做过 cumsum
    int* __restrict__ out_token_idx,        // [total_tokens]
    float* __restrict__ out_weights,        // [total_tokens]
    int seq_len,
    int local_expert_offset
) {
    // 每个 thread 处理 1 个 token
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

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
        }
    }
}


__global__ void moe_permute_copy_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ input,    // [S, 7168]
    const int* __restrict__ permuted_token_idx, // [TotalValidTokens]
    __nv_fp8_e4m3* __restrict__ output,          // [TotalValidTokens, 7168]
    int TotalValidTokens
) {
    const int HIDDEN_DIM = 7168;
    const int VEC_SIZE = 16; // 128 bit / 8 bit = 16 elements per uint4
    
    // 1个 block 处理 1个 token
    int out_row_idx = blockIdx.x; 
    if (out_row_idx >= TotalValidTokens) return;

    int src_row_idx = permuted_token_idx[out_row_idx];

    const uint4* src_ptr4 = reinterpret_cast<const uint4*>(input + src_row_idx * HIDDEN_DIM);
    uint4* dst_ptr4 = reinterpret_cast<uint4*>(output + out_row_idx * HIDDEN_DIM);

    // 256 个 thread
    for (int v = threadIdx.x; v < HIDDEN_DIM / VEC_SIZE; v += blockDim.x) {
        dst_ptr4[v] = src_ptr4[v];
    }
}

void launchCountExpertKernel(
    void* routing_idx,
    void* expert_counts,
    int seq_len,
    int local_expert_offset
) {
    constexpr int threads_per_block = 256;
    const int num_blocks = (seq_len + threads_per_block - 1) / threads_per_block;
    countExpertKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<int*>(expert_counts),
        seq_len,
        local_expert_offset
    );
}

void launchCountExpertAndOffsetsKernel(
    void* routing_idx,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    int seq_len,
    int local_expert_offset
) {
    constexpr int threads_per_block = 256;
    const int num_blocks = (seq_len + threads_per_block - 1) / threads_per_block;
    countExpertKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const int*>(routing_idx),
        static_cast<int*>(expert_counts),
        seq_len,
        local_expert_offset
    );

    exclusiveScan32Kernel<<<1, 32>>>(
        static_cast<const int*>(expert_counts),
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
        seq_len,
        local_expert_offset
    );
}

void launchMoePermuteCopyFp8Kernel(
    void* input,
    void* permuted_token_idx,
    void* output,
    int total_valid_tokens
) {
    constexpr int threads_per_block = 256;
    const int num_blocks = total_valid_tokens;
    moe_permute_copy_fp8_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const __nv_fp8_e4m3*>(input),
        static_cast<const int*>(permuted_token_idx),
        static_cast<__nv_fp8_e4m3*>(output),
        total_valid_tokens
    );
}
"""

permute_cpp_src = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")

void launchCountExpertKernel(
    void* routing_idx,
    void* expert_counts,
    int seq_len,
    int local_expert_offset
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
    int seq_len,
    int local_expert_offset
);

void launchMoePermuteCopyFp8Kernel(
    void* input,
    void* permuted_token_idx,
    void* output,
    int total_valid_tokens
);

torch::Tensor countExpertWrapper(
    torch::Tensor routing_idx,
    int64_t local_expert_offset
) {
    CHECK_CUDA(routing_idx);
    CHECK_CONTIGUOUS(routing_idx);
    CHECK_INT32(routing_idx);
    TORCH_CHECK(routing_idx.dim() == 2, "routing_idx must be [seq_len, 8]");
    TORCH_CHECK(routing_idx.size(1) == 8, "routing_idx second dim must be 8");

    auto expert_counts = torch::zeros(
        {32},
        torch::dtype(torch::kInt32).device(routing_idx.device())
    );

    launchCountExpertKernel(
        routing_idx.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        static_cast<int>(routing_idx.size(0)),
        static_cast<int>(local_expert_offset)
    );

    return expert_counts;
}

std::tuple<torch::Tensor, torch::Tensor, int64_t> countExpertAndOffsetsWrapper(
    torch::Tensor routing_idx,
    int64_t local_expert_offset
) {
    CHECK_CUDA(routing_idx);
    CHECK_CONTIGUOUS(routing_idx);
    CHECK_INT32(routing_idx);
    TORCH_CHECK(routing_idx.dim() == 2, "routing_idx must be [seq_len, 8]");
    TORCH_CHECK(routing_idx.size(1) == 8, "routing_idx second dim must be 8");

    auto expert_counts = torch::zeros(
        {32},
        torch::dtype(torch::kInt32).device(routing_idx.device())
    );
    auto expert_offsets = torch::empty(
        {32},
        torch::dtype(torch::kInt32).device(routing_idx.device())
    );
    auto total_tokens_device = torch::empty(
        {1},
        torch::dtype(torch::kInt32).device(routing_idx.device())
    );

    launchCountExpertAndOffsetsKernel(
        routing_idx.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        total_tokens_device.data_ptr<int>(),
        static_cast<int>(routing_idx.size(0)),
        static_cast<int>(local_expert_offset)
    );

    const int64_t total_tokens = static_cast<int64_t>(
        total_tokens_device.cpu().item<int>()
    );

    return std::make_tuple(expert_counts, expert_offsets, total_tokens);
}

std::tuple<torch::Tensor, torch::Tensor> permuteWrapper(
    torch::Tensor routing_idx,
    torch::Tensor routing_weight,
    torch::Tensor expert_offsets,
    int64_t total_tokens,
    int64_t local_expert_offset
) {
    CHECK_CUDA(routing_idx);
    CHECK_CUDA(routing_weight);
    CHECK_CUDA(expert_offsets);
    CHECK_CONTIGUOUS(routing_idx);
    CHECK_CONTIGUOUS(routing_weight);
    CHECK_CONTIGUOUS(expert_offsets);
    CHECK_INT32(routing_idx);
    CHECK_FLOAT32(routing_weight);
    CHECK_INT32(expert_offsets);

    TORCH_CHECK(routing_idx.dim() == 2, "routing_idx must be [seq_len, 8]");
    TORCH_CHECK(routing_weight.dim() == 2, "routing_weight must be [seq_len, 8]");
    TORCH_CHECK(routing_idx.size(0) == routing_weight.size(0), "routing_idx and routing_weight seq_len must match");
    TORCH_CHECK(routing_idx.size(1) == 8 && routing_weight.size(1) == 8, "routing_idx/routing_weight second dim must be 8");
    TORCH_CHECK(expert_offsets.numel() == 32, "expert_offsets must have 32 elements");
    TORCH_CHECK(total_tokens >= 0, "total_tokens must be non-negative");

    auto out_token_idx = torch::empty(
        {total_tokens},
        torch::dtype(torch::kInt32).device(routing_idx.device())
    );
    auto out_weights = torch::empty(
        {total_tokens},
        torch::dtype(torch::kFloat32).device(routing_idx.device())
    );

    launchPermuteKernel(
        routing_idx.data_ptr<int>(),
        routing_weight.data_ptr<float>(),
        expert_offsets.data_ptr<int>(),
        out_token_idx.data_ptr<int>(),
        out_weights.data_ptr<float>(),
        static_cast<int>(routing_idx.size(0)),
        static_cast<int>(local_expert_offset)
    );

    return std::make_tuple(out_token_idx, out_weights);
}

torch::Tensor moePermuteCopyFp8Wrapper(
    torch::Tensor input,
    torch::Tensor permuted_token_idx
) {
    CHECK_CUDA(input);
    CHECK_CUDA(permuted_token_idx);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(permuted_token_idx);
    CHECK_INT32(permuted_token_idx);

    TORCH_CHECK(input.dim() == 2, "input must be [S, 7168]");
    TORCH_CHECK(input.size(1) == 7168, "input second dim must be 7168");
    TORCH_CHECK(input.element_size() == 1, "input must be 1-byte dtype for fp8 kernel");
    TORCH_CHECK(permuted_token_idx.dim() == 1, "permuted_token_idx must be [TotalValidTokens]");

    const auto total_valid_tokens = permuted_token_idx.size(0);
    auto output = torch::empty(
        {total_valid_tokens, input.size(1)},
        torch::dtype(input.scalar_type()).device(input.device())
    );

    launchMoePermuteCopyFp8Kernel(
        input.data_ptr(),
        permuted_token_idx.data_ptr<int>(),
        output.data_ptr(),
        static_cast<int>(total_valid_tokens)
    );

    return output;
}
"""

# JIT 编译
# my_lib = load(
#     name="fused_gating",
#     sources=["wrapper.cpp", "fused_gating.cu", "permute.cu"],
#     verbose=True
# )

my_gating = load_inline(
    name = "fused_gating",
    cuda_sources=fused_gating_src,
    cpp_sources=fused_gating_cpp_src,
    functions=["fusedGatingWrapper"],
    verbose=True
)

my_permute = load_inline(
    name = "permute",
    cuda_sources=permute_src,
    cpp_sources=permute_cpp_src,
    functions=["countExpertWrapper", "permuteWrapper", "moePermuteCopyFp8Wrapper", "countExpertAndOffsetsWrapper"]
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

def torch_impl_gating(routing_logits, routing_bias, routing_scaling_factor):
    H = 7168
    I = 2048
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168" 
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4


    logits = routing_logits.to(torch.float32)                      # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)              # [E_global]

    # Sigmoid
    s = 1.0 / (1.0 + torch.exp(-logits))                       # [T, E]
    s_with_bias = s + bias                                     # [T, E] (broadcast)
    # print(s_with_bias)

    # Grouping
    group_size = E_global // N_GROUP # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)    # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)                        # [T, 8]
    # print(f"group_scores: {top2_vals}")

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  # [T, 4]
    # print(f"group_idx: {group_idx}")
    group_mask = torch.zeros_like(group_scores)                # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)  # [T, E]

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)                  # [T, E]
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=True)  # [T, 8]
    # print(s_with_bias[0][71])

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)                                    # [T, E]
    M.scatter_(1, topk_idx, 1.0)                               # 0/1 mask
    weights = s * M                                            # [T, E]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    # print(weights)
    weights = (weights / weights_sum) * routing_scaling_factor  # [T, E]

    topk_weights = torch.gather(weights, 1, topk_idx)
    return topk_idx, topk_weights

def fused_impl_gating(routing_logits, routing_bias, routing_scaling_factor):
    routing_idx, routing_weights = my_gating.fusedGatingWrapper(routing_logits, routing_bias, routing_scaling_factor)
    return routing_idx, routing_weights

def test_time(impl_func, *args):
    # warmup
    for _ in range(20):
        impl_func(*args)

    n_iters = 100
    start_time = time.time()
    for _ in range(n_iters):
        impl_func(*args)
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iters
    print(f"Average execution time over {n_iters} runs: {avg_time:.6f} seconds")

def test_gating():
    print("========== test ==========")
    torch.manual_seed(42)
    torch.set_default_device('cuda:7')

    seq_len = 20  # 示例序列长度
    num_experts = 256
    num_local_experts = 32  # 假设当前 Rank 负责的专家数
    hidden_size = 7168
    intermediate_size = 2048
    gemm1_out_size = 4096

    block_size = 128
    num_hidden_blocks = hidden_size // block_size
    num_intermediate_blocks = intermediate_size // block_size
    num_gemm1_out_blocks = gemm1_out_size // block_size


    routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32)
    routing_bias = torch.randn(num_experts, dtype=torch.bfloat16)
    hidden_states = torch.randn(seq_len, hidden_size).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.randn(num_hidden_blocks, seq_len, dtype=torch.float32)
    gemm1_weights = torch.randn(
        num_local_experts, gemm1_out_size, hidden_size
    ).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.randn(
        num_local_experts, num_gemm1_out_blocks, num_hidden_blocks, dtype=torch.float32
    )
    gemm2_weights = torch.randn(
        num_local_experts, hidden_size, intermediate_size
    ).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.randn(
        num_local_experts, num_hidden_blocks, num_intermediate_blocks, dtype=torch.float32
    )
    local_expert_offset = 32
    routed_scaling_factor = 1.11

    print("============ torch impl =============")
    torch_idx, torch_weights = torch_impl_gating(routing_logits, routing_bias, routed_scaling_factor)
    print("Torch topk indices shape:", torch_idx.shape)
    print("Torch weights shape:", torch_weights.shape)

    print("============ fused impl =============")
    fused_idx, fused_weights = fused_impl_gating(routing_logits, routing_bias, routed_scaling_factor)
    print("Fused topk indices shape:", fused_idx.shape)
    print("Fused weights shape:", fused_weights.shape)


    test_time(torch_impl_gating, routing_logits, routing_bias, routed_scaling_factor)
    test_time(fused_impl_gating, routing_logits, routing_bias, routed_scaling_factor)
    print((torch_idx == fused_idx).all())
    print(torch.allclose(torch_weights, fused_weights, atol=1e-6))
    print("==================")
    print(f"torch indices is {torch_idx}")
    print(f"fused indices is {fused_idx}")

    def get_experts_cnt(routing_idx, local_expert_offset):
        counts = []
        for i in range(32):
            mask = (routing_idx == (local_expert_offset + i))
            count = mask.sum().item()
            counts.append(count)
        
        return torch.tensor(counts, device=routing_idx.device)
    print("torch local expert counts:", get_experts_cnt(torch_idx, local_expert_offset))
    print("fused local expert counts:", get_experts_cnt(fused_idx, local_expert_offset))

def torch_impl_permute(
    routing_idx, 
    routing_weights,
    local_expert_offset
):
    def count(routing_idx, local_expert_offset):
        counts = []
        for i in range(32):
            mask = (routing_idx == (local_expert_offset + i))
            count = mask.sum().item()
            counts.append(count)
        
        return torch.tensor(counts, device=routing_idx.device)
    
    def permute(
        routing_idx,
        routing_weights,
        local_expert_offset,
        counts
    ):
        """
        Args:
            routing_idx: [seq_len, 8]
            routing_weights: [seq_len, 8]
            local_expert_offset: int
            counts: [32] - 预先计算好的每个 local expert 的 token 总数
        """
        num_local_experts = counts.size(0)
        device = routing_idx.device
        
        # 1. 筛选出本地 Expert 相关的掩码
        is_local = (routing_idx >= local_expert_offset) & (routing_idx < local_expert_offset + num_local_experts)
        
        # 2. 提取有效数据并转为本地索引 [0, 31]
        valid_expert_ids = (routing_idx[is_local] - local_expert_offset).long()
        valid_weights = routing_weights[is_local]
        
        # 生成原始 token 索引 (0 到 seq_len-1)
        token_indices = torch.arange(routing_idx.size(0), device=device).view(-1, 1).expand_as(routing_idx)
        valid_token_ids = token_indices[is_local]

        # 3. 计算每个 token 在其所属 expert 组内的偏移量 (Rank)
        # 例如：如果 valid_expert_ids 是 [0, 1, 0, 2]，那么对应的 rank 是 [0, 0, 1, 0]
        # 我们通过对每个 expert ID 进行累加计数来实现
        # 这里使用一个小技巧：对 one-hot 后的结果做 cumsum
        one_hot = torch.nn.functional.one_hot(valid_expert_ids, num_classes=num_local_experts)
        expert_rank = torch.cumsum(one_hot, dim=0) * one_hot
        expert_rank = expert_rank.sum(dim=-1) - 1 # 减 1 变为从 0 开始的索引

        # 4. 计算每个 Expert 在输出 tensor 中的起始全局偏移量
        # cumulative_offsets[i] 表示第 i 个 expert 在 permute_token_idx 中的开始位置
        expert_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])

        # 5. 计算最终在 permute_token_idx 中的位置
        # 最终位置 = 该 Expert 的起始位置 + 该 Token 在 Expert 内部的序号
        final_positions = expert_offsets[valid_expert_ids] + expert_rank

        # 6. 放置数据
        num_total_valid = counts.sum().item()
        permute_token_idx = torch.empty(num_total_valid, dtype=torch.long, device=device)
        permute_weight = torch.empty(num_total_valid, dtype=routing_weights.dtype, device=device)

        permute_token_idx.scatter_(0, final_positions, valid_token_ids)
        permute_weight.scatter_(0, final_positions, valid_weights)

        return permute_token_idx, permute_weight

    expert_cnts = count(routing_idx, local_expert_offset)
    print("torch Expert counts:", expert_cnts)

    permute_token_idx, permute_weight = permute(routing_idx, routing_weights, local_expert_offset, expert_cnts)
    
    return expert_cnts, permute_token_idx, permute_weight

def fused_impl_permute(
    routing_idx, 
    routing_weights,
    local_expert_offset
):
    expert_cnts, offsets, total_valid_tokens = my_permute.countExpertAndOffsetsWrapper(routing_idx, local_expert_offset)
    permute_token_idx, permute_weight = my_permute.permuteWrapper(routing_idx, routing_weights, offsets, total_valid_tokens, local_expert_offset)
    return expert_cnts, permute_token_idx, permute_weight

def torch_copy_impl(
    hidden_states,
    permute_token_idx,
):
    # hidden_states: [seq_len, hidden_dim]
    # permute_token_idx: [num_valid_tokens]
    permute_hidden_states = hidden_states[permute_token_idx]
    return permute_hidden_states

def fused_copy_impl(
    hidden_states,
    permute_token_idx,
):
    permute_hidden_states = my_permute.moePermuteCopyFp8Wrapper(hidden_states, permute_token_idx)
    return permute_hidden_states


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
    # print("\n*************** fused start *****************")
    # 1. gating
    routing_idx, routing_weights = fused_impl_gating(routing_logits, routing_bias, routed_scaling_factor)
    # print(routing_idx)
    # print(routing_weights)

    # 2. permute
    expert_cnts, permute_token_idx, permute_weight = fused_impl_permute(routing_idx, routing_weights, local_expert_offset)
    counts_cpu = expert_cnts.to(device="cpu", dtype=torch.int64).tolist()
    if sum(counts_cpu) == 0:
        print("No tokens routed to local experts.")
        return torch.zeros_like(hidden_states)

    # 3. copy
    # print(f"Expert counts: {expert_cnts}")
    # print(f"Permute token indices: {permute_token_idx.shape}")
    # print(f"Permute weights: {permute_weight.shape}")
    # return
    # permute_hidden_states = fused_copy_impl(hidden_states, permute_token_idx)
    permute_hidden_states = torch_copy_impl(hidden_states, permute_token_idx)

    # 4. gemm1 & activation
    num_experts = gemm1_weights.shape[0]
    offsets_cpu = [0] * num_experts
    running = 0
    for e, cnt in enumerate(counts_cpu):
        offsets_cpu[e] = running
        running += cnt

    permute_hidden_states_list = []
    permute_hidden_states_scale_list = []
    permute_weight_list = []
    permute_token_idx_list = []
    gemm1_weights_list = []
    gemm1_weights_scale_list = []
    gemm2_weights_list = []
    gemm2_weights_scale_list = []

    for e, s_i in enumerate(counts_cpu):
        if s_i == 0:
            continue
        start = offsets_cpu[e]
        end = start + s_i
        # print(f"Expert {e}: token count = {s_i}")
        permute_hidden_states_list.append(permute_hidden_states[start:end])
        permute_weight_list.append(permute_weight[start:end])
        token_idx = permute_token_idx[start:end]
        permute_token_idx_list.append(token_idx)
        permute_hidden_states_scale_list.append(hidden_states_scale.index_select(1, token_idx).contiguous())
        gemm1_weights_list.append(gemm1_weights[e])
        gemm1_weights_scale_list.append(gemm1_weights_scale[e])
        gemm2_weights_list.append(gemm2_weights[e])
        gemm2_weights_scale_list.append(gemm2_weights_scale[e])

    # G1 = permute_hidden_states_list[0] @ gemm1_weights_list[0].t()
    # X1 = G1[:, :I]                                         # [Tk, I]
    # X2 = G1[:, I:]                                         # [Tk, I]
    # silu_X2 = X2 / (1.0 + torch.exp(-X2))                  # [Tk, I]
    # C = silu_X2 * X1
    # return C
    gemm1_output_list, gemm1_output_scale_list = gemm1(
        permute_hidden_states_list,
        permute_hidden_states_scale_list,
        gemm1_weights_list,
        gemm1_weights_scale_list,
    )
    # return gemm1_output_fp8_list[0]

    # print(gemm1_output_list[0][0][:10])
    # print(gemm1_output_scale_list[0])
    # return gemm1_output_list[0][0], gemm1_output_scale_list[0]
    
    # 5. gemm2 & output
    output = gemm2(
        gemm1_output_list,
        # gemm1_output_scale_list,
        gemm2_weights_list,
        gemm2_weights_scale_list,
        permute_weight_list,
        permute_token_idx_list,
        seq_len=hidden_states.size(0)
    )

    # print("*************** fused over *****************\n")
    return output

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

# print("========== test ==========")
# torch.manual_seed(42)
# torch.set_default_device('cuda:0')

# seq_len = 1 
# num_experts = 256
# num_local_experts = 32
# hidden_size = 7168
# intermediate_size = 2048
# gemm1_out_size = 4096

# block_size = 128
# num_hidden_blocks = hidden_size // block_size
# num_intermediate_blocks = intermediate_size // block_size
# num_gemm1_out_blocks = gemm1_out_size // block_size


# routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32)
# routing_bias = torch.randn(num_experts, dtype=torch.bfloat16)
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
# local_expert_offset = 32
# routed_scaling_factor = 1.11


# test_time(fused_moe, routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor)
# test_time(run, routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor)

# # time.sleep(0.5)

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

# print(f"fused: {fused_output[0][:10]}, {fused_output.dtype}")
# print(f"std: {std_output[0][:10]}, {std_output.dtype}")


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
