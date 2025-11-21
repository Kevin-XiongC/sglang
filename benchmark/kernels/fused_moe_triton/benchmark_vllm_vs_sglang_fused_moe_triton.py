# python3 benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py --model /DeepSeek-V3/ --tp-size 8 --use-fp8-w8a8
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import triton
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts as fused_moe_vllm,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe as fused_moe_sglang,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig, select_experts
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
from common_utils import get_model_config


@dataclass
class MoEInputs:
    """Container for MoE inputs and configuration."""
    x: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    topk_output: StandardTopKOutput
    w1_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a1_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[Tuple[int, int]] = None
    # For cutlass_w4a8_moe
    cutlass_inputs: Optional[Tuple] = None


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    """Pack interleaved int4 values into int8."""
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "the last dim size of int4_values_interleaved tensor must be even."
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)
    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]
    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)
    return packed_tensor.to(torch.int8)


def pack_interleave(
    num_experts: int,
    ref_weight: torch.Tensor,
    ref_scale: torch.Tensor,
    alignment: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack and interleave weights and scales for W4A8 quantization."""
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8).contiguous()

    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0],
        ref_scale.shape[1],
        (ref_scale.shape[2] // alignment),
        alignment,
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        ref_scale.shape[0],
        ref_scale.shape[2] // alignment,
        ref_scale.shape[1] * alignment,
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()
    return w_q, w_scale


def prepare_fused_moe_cutlass_w4a8_input(
    x: torch.Tensor,
    model_config: Dict,
    tp_size: int = 4,
    group_size: int = 128,
) -> Tuple:
    """Prepare inputs for cutlass W4A8 MoE."""
    num_experts = model_config["num_experts"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    device = x.device

    N = shard_intermediate_size // 2
    K = hidden_size

    # Generate random quantized weights
    ref_weight_1 = torch.randint(
        -8, 8, (num_experts, N * 2, K), dtype=torch.int8, device=device
    )
    ref_weight_2 = torch.randint(
        -8, 8, (num_experts, K, N), dtype=torch.int8, device=device
    )

    # Generate scales
    affine_coeff = 0.005
    scale_1 = (
        torch.randn(num_experts, N * 2, K // group_size, dtype=dtype, device=device)
        * affine_coeff
    )
    scale_2 = (
        torch.randn(num_experts, K, N // group_size, dtype=dtype, device=device)
        * affine_coeff
    )

    w1_q, w1_scale = pack_interleave(num_experts, ref_weight_1, scale_1)
    w2_q, w2_scale = pack_interleave(num_experts, ref_weight_2, scale_2, alignment=1)

    # Prepare strides
    a_strides1 = torch.full((num_experts, 3), K, device=device, dtype=torch.int64)
    c_strides1 = torch.full((num_experts, 3), 2 * N, device=device, dtype=torch.int64)
    a_strides2 = torch.full((num_experts, 3), N, device=device, dtype=torch.int64)
    c_strides2 = torch.full((num_experts, 3), K, device=device, dtype=torch.int64)
    b_strides1 = a_strides1
    s_strides13 = c_strides1
    b_strides2 = a_strides2
    s_strides2 = c_strides2

    expert_map = torch.arange(num_experts, dtype=torch.int32, device=device)

    return (
        x,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        expert_map,
    )

def create_quant_config(
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[Tuple[int, int]] = None,
    per_act_token_quant: bool = True,
) -> FusedMoEQuantConfig:
    """Create quantization config for vLLM."""
    return FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
    )


def create_strides(
    num_experts: int,
    hidden_size: int,
    shard_intermediate_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create stride tensors for cutlass MoE."""
    ab_strides1 = torch.full((num_experts,), hidden_size, device=device, dtype=torch.int64)
    ab_strides2 = torch.full((num_experts,), shard_intermediate_size, device=device, dtype=torch.int64)
    c_strides1 = torch.full((num_experts,), 2 * shard_intermediate_size, device=device, dtype=torch.int64)
    c_strides2 = torch.full((num_experts,), hidden_size, device=device, dtype=torch.int64)
    return ab_strides1, ab_strides2, c_strides1, c_strides2


def fused_moe_cutlass_w4a8_api(
    inputs: MoEInputs,
    model_config: Dict,
) -> torch.Tensor:
    """Call cutlass W4A8 MoE API."""
    cutlass_inputs = inputs.cutlass_inputs
    if cutlass_inputs is None:
        raise ValueError("cutlass_inputs must be provided for cutlass_w4a8_moe")

    (
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        expert_map,
    ) = cutlass_inputs

    topk_weights = inputs.topk_output.topk_weights
    topk_ids = expert_map[inputs.topk_output.topk_ids]
    device = a.device

    # Initialize expert offsets and problem sizes if needed
    num_experts = model_config["num_experts"]
    expert_offsets = torch.empty((num_experts + 1), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)

    # Generate activation scales if not provided
    a1_scale = inputs.a1_scale or torch.randn(1, dtype=torch.float32, device=device)
    a2_scale = inputs.a2_scale or torch.randn(1, dtype=torch.float32, device=device)

    return cutlass_w4a8_moe(
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a1_scale,
        a2_scale,
        apply_router_weight_on_input=False,
    )


def fused_moe_cutlass_api(
    inputs: MoEInputs,
    model_config: Dict,
) -> torch.Tensor:
    """Call cutlass FP8 MoE API."""
    num_experts = model_config["num_experts"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    device = inputs.x.device

    ab_strides1, ab_strides2, c_strides1, c_strides2 = create_strides(
        num_experts, hidden_size, shard_intermediate_size, device
    )

    quant_config = create_quant_config(
        inputs.w1_scale,
        inputs.w2_scale,
        inputs.a1_scale,
        inputs.a2_scale,
        inputs.block_shape,
        per_act_token_quant=True,
    )

    return cutlass_moe_fp8(
        a=inputs.x,
        w1_q=inputs.w1,
        w2_q=inputs.w2,
        topk_weights=inputs.topk_output.topk_weights,
        topk_ids=inputs.topk_output.topk_ids,
        ab_strides1=ab_strides1,
        ab_strides2=ab_strides2,
        c_strides1=c_strides1,
        c_strides2=c_strides2,
        quant_config=quant_config,
        activation="silu",
        global_num_experts=num_experts,
    )


def fused_moe_vllm_api(
    inputs: MoEInputs,
    model_config: Dict,
) -> torch.Tensor:
    """Call vLLM fused MoE API."""
    quant_config = create_quant_config(
        inputs.w1_scale,
        inputs.w2_scale,
        inputs.a1_scale,
        inputs.a2_scale,
        inputs.block_shape,
        per_act_token_quant=True,
    )

    return fused_moe_vllm(
        inputs.x,
        inputs.w1,
        inputs.w2,
        topk_weights=inputs.topk_output.topk_weights,
        topk_ids=inputs.topk_output.topk_ids,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
    )


def fused_moe_sglang_api(
    inputs: MoEInputs,
    model_config: Dict,
) -> torch.Tensor:
    """Call SGLang fused MoE API."""
    return fused_moe_sglang(
        inputs.x,
        inputs.w1,
        inputs.w2,
        topk_output=inputs.topk_output,
        moe_runner_config=MoeRunnerConfig(
            apply_router_weight_on_input=False, inplace=True
        ),
        use_fp8_w8a8=bool(inputs.w1_scale is not None),
        w1_scale=inputs.w1_scale,
        w2_scale=inputs.w2_scale,
        a1_scale=inputs.a1_scale,
        a2_scale=inputs.a2_scale,
        per_channel_quant=True,
        block_shape=inputs.block_shape,
    )


def prepare_weights_and_scales(
    num_experts: int,
    hidden_size: int,
    shard_intermediate_size: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    block_shape: Optional[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Prepare weight tensors and scales for MoE."""
    if use_fp8_w8a8:
        init_dtype = dtype
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device=device
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype, device=device
        )
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

        if block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32, device=device
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32, device=device
            )
        return w1, w2, w1_scale, w2_scale
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=dtype, device=device
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=dtype, device=device
        )
        return w1, w2, None, None


# Provider function mapping
PROVIDER_FUNCTIONS: Dict[str, Callable] = {
    "vllm_fused_moe_triton": fused_moe_vllm_api,
    "sglang_fused_moe_triton": fused_moe_sglang_api,
    # "cutlass_moe_fp8": fused_moe_cutlass_api,
    # "cutlass_w4a8_moe": fused_moe_cutlass_w4a8_api,
}


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=list(range(2, 256, 2)),
        line_arg="provider",
        line_vals=list(PROVIDER_FUNCTIONS.keys()),
        line_names=list(PROVIDER_FUNCTIONS.keys()),
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("red", "-"),
            ("purple", "-"),
        ],
        ylabel="Time (ms)",
        plot_name="fused-moe-performance",
        args={},
    )
)
def benchmark(batch_size, provider, model_config, use_fp8_w8a8=False):
    """Benchmark MoE implementations."""
    print(f"benchmark {provider} with batch_size={batch_size}")
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    num_tokens = batch_size
    num_experts = model_config["num_experts"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    topk = model_config["topk"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]
    device = torch.device("cuda")

    # Prepare inputs
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w1, w2, w1_scale, w2_scale = prepare_weights_and_scales(
        num_experts, hidden_size, shard_intermediate_size, dtype, use_fp8_w8a8, block_shape, device
    )

    # Prepare topk output
    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_config = TopKConfig(top_k=topk, renormalize=True)
    topk_output = select_experts(x, input_gating, topk_config)

    # Prepare inputs container
    inputs = MoEInputs(
        x=x,
        w1=w1,
        w2=w2,
        topk_output=topk_output,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    # Special handling for cutlass_w4a8_moe
    if provider == "cutlass_w4a8_moe":
        inputs.cutlass_inputs = prepare_fused_moe_cutlass_w4a8_input(x, model_config)

    # Get API function
    api_func = PROVIDER_FUNCTIONS[provider]

    # Warmup
    for _ in range(10):
        _ = api_func(inputs, model_config)
    torch.cuda.synchronize()

    # Benchmark
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        lambda: api_func(inputs, model_config),
        quantiles=quantiles,
    )
    return ms, min_ms, max_ms


def main():
    """Main entry point for benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark fused MoE implementations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model name or path",
    )
    parser.add_argument(
        "--tp-size", "--tp", type=int, default=2, help="Tensor parallel size"
    )
    parser.add_argument(
        "--ep-size", "--ep", type=int, default=1, help="Expert parallel size"
    )
    parser.add_argument(
        "--use-fp8-w8a8", action="store_true", help="Use FP8 W8A8 quantization"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/vllm_sglang_fused_moe/",
        help="Path to save benchmark results",
    )
    args = parser.parse_args()

    try:
        model_config = get_model_config(args.model, args.tp_size, args.ep_size)
        benchmark.run(
            show_plots=True,
            print_data=True,
            save_path=args.save_path,
            model_config=model_config,
            use_fp8_w8a8=args.use_fp8_w8a8,
        )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
