SGLANG_SET_CPU_AFFINITY=1 SGL_ENABLE_JIT_DEEPGEMM=1 \
LD_LIBRARY_PATH=/root/nccl-rdma-sharp-plugins/src/.libs:/root/nccl/build/lib:${LD_LIBRARY_PATH} \
PYTHONUNBUFFERED=1 MC_TE_METRIC=true GLOO_SOCKET_IFNAME=eth0 NCCL_NET_GDR_LEVEL=2 \
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 \
NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1 \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" TORCH_CUDA_ARCH_LIST=9.0 \
    python3 -m sglang.launch_server \
    --model-path /root/Llama-2-70b-hf --served-model-name llama --trust-remote-code \
    --host 0.0.0.0 --watchdog-timeout 1000000 \
    --tokenizer-worker-num 4 --disaggregation-mode prefill \
    --disaggregation-ib-device mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
    --context-length 4096 --max-prefill-tokens 32000 --chunked-prefill-size 32128 \
    --disable-radix-cache --tp-size 8 \
    --page-size 64 --max-running-requests 1024 --mem-fraction-static 0.90 \
    --load-balance-method round_robin  --load-format dummy 


SGLANG_SET_CPU_AFFINITY=1 SGL_ENABLE_JIT_DEEPGEMM=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
LD_LIBRARY_PATH=/root/nccl-rdma-sharp-plugins/src/.libs:/root/nccl/build/lib:${LD_LIBRARY_PATH} \
PYTHONUNBUFFERED=1 MC_TE_METRIC=true GLOO_SOCKET_IFNAME=eth0 NCCL_NET_GDR_LEVEL=2 \
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 \
NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1 \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" TORCH_CUDA_ARCH_LIST=9.0 \
    python3 -m sglang.launch_server \
    --model-path /root/Llama-2-70b-hf --served-model-name llama \
    --trust-remote-code --tokenizer-worker-num 4 \
    --disaggregation-mode decode --disaggregation-ib-device mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
    --disable-radix-cache --enable-dp-lm-head --enable-dp-attention \
    --tp-size 8 --dp-size 8 \
    --page-size 64 --watchdog-timeout 1000000 --host 0.0.0.0  --chunked-prefill-size 6144 \
    --prefill-round-robin-balance --load-format dummy --mem-fraction-static 0.7  --max-running-requests 4096


python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --policy round_robin \
  --prefill http://10.96.191.142:30000 \
  --decode http://10.96.191.143:30000 \
  --host 0.0.0.0 --port 8010 \
  --max-concurrent-requests 2048