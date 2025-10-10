"""
Mooncake RDMA KV cache block transfer micro-benchmark (two machines, CUDA only).

Prerequisites:
- Install Mooncake per docs: https://github.com/kvcache-ai/Mooncake
- Install PyTorch with CUDA support
- Ensure RDMA/IB is properly configured and both machines can reach each other.

Roles:
- server: initializes Mooncake engine, allocates and registers 61-layer KV cache buffers,
          each layer divided into blocks, sends layer pointers and block info via ZMQ.
- client: receives KV cache metadata via ZMQ, randomly selects N blocks from any layers
          and transfers them for specified duration.

Examples:
Server (machine A):
  python -m sglang.tools.mooncake_rdma_bw \
    --role server \
    --host 10.0.0.1 \
    --gpu-id 0 \
    --ib-device mlx5_0 \
    --kv-layers 61 \
    --layer-size-bytes 67108864 \
    --blocks-per-layer 64 \
    --zmq-push-bind tcp://*:5555

Client (machine B):
  python -m sglang.tools.mooncake_rdma_bw \
    --role client \
    --host 10.0.0.2 \
    --gpu-id 0 \
    --ib-device mlx5_0 \
    --block-size-bytes 1048576 \
    --random-blocks 20 \
    --duration 10 \
    --zmq-pull-connect tcp://10.0.0.1:5555

Notes:
- `--host` should be the local IP/hostname of each machine that peers can reach.
- This script only supports CUDA buffers (requires PyTorch).
- Server allocates `--kv-layers` buffers, each divided into `--blocks-per-layer` blocks.
- Client randomly selects `--random-blocks` blocks from any layers for each transfer.
- ZMQ handshake is required to share KV cache metadata.
"""

import argparse
import random
import sys
import time
from typing import Optional, Tuple, List
import json

import torch

try:
    import zmq  # Optional, only used when ZMQ options provided
except Exception:  # pragma: no cover
    zmq = None  # type: ignore

from sglang.srt.disaggregation.mooncake.transfer_engine import (
    MooncakeTransferEngine,
)


def allocate_cuda_buffer(size_bytes: int) -> Tuple[int, torch.Tensor]:
    """Allocate CUDA buffer and return (ptr, tensor)."""
    num_elems = (size_bytes + 3) // 4
    tensor = torch.empty(num_elems, dtype=torch.int32, device="cuda")
    ptr = tensor.data_ptr()
    return ptr, tensor


def human_bandwidth(bytes_per_sec: float) -> str:
    gbps = (bytes_per_sec * 8.0) / 1e9
    gBps = bytes_per_sec / 1e9
    return f"{gBps:.2f} GB/s, {gbps:.2f} Gbps"


def run_server(
    host: str,
    gpu_id: int,
    ib_device: Optional[str],
    kv_layers: int,
    layer_size_bytes: int,
    blocks_per_layer: int,
    zmq_push_bind: Optional[str],
):
    engine = MooncakeTransferEngine(hostname=host, gpu_id=gpu_id, ib_device=ib_device)

    # Allocate KV cache layers
    layer_ptrs: List[int] = []
    layer_holders: List[torch.Tensor] = []
    for layer_idx in range(kv_layers):
        ptr, holder = allocate_cuda_buffer(layer_size_bytes)
        engine.register(ptr, layer_size_bytes)
        layer_ptrs.append(ptr)
        layer_holders.append(holder)
        print(f"Layer {layer_idx}: ptr={ptr}, size={layer_size_bytes/1024:.1f}KB, blocks={blocks_per_layer}")

    block_size_bytes = layer_size_bytes // blocks_per_layer
    print("Mooncake RDMA Server ready")
    print(f"session_id={engine.session_id}")
    print(f"kv_layers={kv_layers}")
    print(f"layer_size_bytes={layer_size_bytes/1024:.1f}KB")
    print(f"blocks_per_layer={blocks_per_layer}")
    print(f"block_size_bytes={block_size_bytes/1024:.1f}KB")

    # Send KV cache metadata via ZMQ
    if zmq_push_bind is not None:
        if zmq is None:
            raise RuntimeError("pyzmq is required when --zmq-push-bind is specified")
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUSH)
        socket.bind(zmq_push_bind)
        md = {
            "session_id": engine.session_id,
            "kv_layers": kv_layers,
            "layer_size_bytes": layer_size_bytes,
            "blocks_per_layer": blocks_per_layer,
            "block_size_bytes": block_size_bytes,
            "layer_ptrs": layer_ptrs,
        }
        socket.send_string(json.dumps(md))
        print(f"ZMQ PUSH sent KV cache metadata to {zmq_push_bind}")

    print("Keep this process running while client is benchmarking...")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def run_client(
    host: str,
    gpu_id: int,
    ib_device: Optional[str],
    random_blocks: int,
    duration_sec: float,
    warmup_iters: int,
    report_interval_sec: float,
    zmq_pull_connect: Optional[str],
    use_random_blocks: bool,
):
    # Get KV cache metadata via ZMQ
    if zmq_pull_connect is None:
        raise ValueError("--zmq-pull-connect is required for client role")
    if zmq is None:
        raise RuntimeError("pyzmq is required when --zmq-pull-connect is specified")
    
    context = zmq.Context.instance()
    socket = context.socket(zmq.PULL)
    socket.connect(zmq_pull_connect)
    msg = socket.recv_string()
    try:
        md = json.loads(msg)
    except Exception as e:
        raise RuntimeError(f"Failed to parse ZMQ metadata: {e}")
    
    session_id = md["session_id"]
    kv_layers = md["kv_layers"]
    layer_size_bytes = md["layer_size_bytes"]
    blocks_per_layer = md["blocks_per_layer"]
    block_size_bytes = md["block_size_bytes"]
    layer_ptrs = [int(x) for x in md["layer_ptrs"]]
    
    print(f"Received KV cache metadata: {kv_layers} layers, {layer_size_bytes/1024:.1f}KB each")
    print(f"Blocks per layer: {blocks_per_layer}, block size: {block_size_bytes/1024:.1f}KB")
    print(f"Random blocks per transfer: {random_blocks}")

    engine = MooncakeTransferEngine(hostname=host, gpu_id=gpu_id, ib_device=ib_device)

    # Allocate source buffer for transfers
    src_ptr, src_holder = allocate_cuda_buffer(block_size_bytes)
    engine.register(src_ptr, block_size_bytes)
    
    # Initialize source buffer
    src_holder.fill_(0x7F7F7F7F)
    torch.cuda.synchronize()

    def get_random_blocks():
        """Generate random block selections across all layers."""
        selected_blocks = []
        for layer_idx in range(kv_layers):
            for _ in range(random_blocks):
                block_idx = random.randint(0, blocks_per_layer - 1) if use_random_blocks else list(range(random_blocks))
                layer_ptr = layer_ptrs[layer_idx]
                block_offset = block_idx * block_size_bytes
                block_ptr = layer_ptr + block_offset
                selected_blocks.append((layer_idx, block_idx, block_ptr))
        return selected_blocks

    # Warmup
    for _ in range(max(0, warmup_iters)):
        blocks = get_random_blocks()
        for layer_idx, block_idx, block_ptr in blocks:
            ret = engine.transfer_sync(session_id, src_ptr, block_ptr, block_size_bytes)
            if ret < 0:
                raise RuntimeError("Warmup transfer failed")

    torch.cuda.synchronize()

    # Timed loop with random block selection
    total_bytes = 0
    transfer_count = 0
    t0 = time.perf_counter()
    last_report = t0
    
    print("Starting random KV cache block transfers...")
    while True:
        # Randomly select N blocks from any layers
        blocks = get_random_blocks()
        
        for layer_idx, block_idx, block_ptr in blocks:
            ret = engine.transfer_sync(session_id, src_ptr, block_ptr, block_size_bytes)
            if ret < 0:
                raise RuntimeError("transfer_sync failed")
        
        total_bytes += block_size_bytes * random_blocks
        transfer_count += 1

        now = time.perf_counter()
        if now - t0 >= duration_sec:
            break
        if report_interval_sec > 0 and (now - last_report) >= report_interval_sec:
            bps = total_bytes / (now - t0)
            print(f"[progress] {human_bandwidth(bps)} over {now - t0:.2f}s, transfers={transfer_count}")
            last_report = now

    torch.cuda.synchronize()

    t1 = time.perf_counter()
    elapsed = max(1e-9, t1 - t0)
    bps = total_bytes / elapsed

    print("Mooncake RDMA Client result")
    print(f"block_size_bytes={block_size_bytes/1024:.1f}KB")
    print(f"random_blocks_per_transfer={random_blocks}")
    print(f"kv_layers={kv_layers}")
    print(f"blocks_per_layer={blocks_per_layer}")
    print(f"duration={elapsed:.3f}s")
    print(f"transfers={transfer_count}")
    print(f"total_blocks_transferred={transfer_count * random_blocks}")
    print(f"bytes_transferred={total_bytes}")
    print(f"bandwidth={human_bandwidth(bps)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mooncake RDMA KV cache block transfer benchmark")
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--host", type=str, required=True, help="Local host/IP")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--ib-device", type=str, default=None)
    
    # Server arguments
    parser.add_argument("--kv-layers", type=int, default=61, help="Number of KV cache layers to allocate (server only)")
    parser.add_argument("--layer-size-bytes", type=int, default=64 * 1024 * 1024, help="Size of each KV cache layer in bytes (server only)")
    parser.add_argument("--blocks-per-layer", type=int, default=64, help="Number of blocks per layer (server only)")
    parser.add_argument("--zmq-push-bind", type=str, default=None, help="ZMQ PUSH bind address, e.g. tcp://*:5555 (server only)")
    
    # Client arguments
    parser.add_argument("--random-blocks", type=int, default=20, help="Number of random blocks to transfer per iteration (client only)")
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds (client only)")
    parser.add_argument("--zmq-pull-connect", type=str, default=None, help="ZMQ PULL connect address, e.g. tcp://10.0.0.1:5555 (client only)")
    parser.add_argument("--warmup-iters", type=int, default=3, help="Number of warmup iterations (client only)")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Progress report interval in seconds (client only)")
    parser.add_argument("--use-random-blocks", type=bool, action="store_true", help="Use random blocks to transfer per iteration (client only)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.role == "server":
        if args.zmq_push_bind is None:
            print("--zmq-push-bind is required for server role", file=sys.stderr)
            sys.exit(2)
        run_server(
            host=args.host,
            gpu_id=args.gpu_id,
            ib_device=args.ib_device,
            kv_layers=args.kv_layers,
            layer_size_bytes=args.layer_size_bytes,
            blocks_per_layer=args.blocks_per_layer,
            zmq_push_bind=args.zmq_push_bind,
        )
        return

    # client
    if args.zmq_pull_connect is None:
        print("--zmq-pull-connect is required for client role", file=sys.stderr)
        sys.exit(2)

    run_client(
        host=args.host,
        gpu_id=args.gpu_id,
        ib_device=args.ib_device,
        random_blocks=args.random_blocks,
        duration_sec=args.duration,
        warmup_iters=args.warmup_iters,
        report_interval_sec=args.report_interval,
        zmq_pull_connect=args.zmq_pull_connect,
        use_random_blocks=args.use_random_blocks,
    )


if __name__ == "__main__":
    main()


