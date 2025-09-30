"""
Mooncake RDMA point-to-point bandwidth micro-benchmark (two machines).

Prerequisites:
- Install Mooncake per docs: https://github.com/kvcache-ai/Mooncake
- Ensure RDMA/IB is properly configured and both machines can reach each other.

Roles:
- server: initializes Mooncake engine, allocates and registers a receive buffer,
          prints `session_id` and `peer_addr` (the receive buffer address) for the client.
- client: initializes Mooncake engine, allocates and registers a source buffer,
          repeatedly `transfer_sync` to `peer_addr` for `duration` seconds and reports bandwidth.

Examples:
Server (machine A):
  python -m sglang.tools.mooncake_rdma_bw \
    --role server \
    --host 10.0.0.1 \
    --gpu-id 0 \
    --ib-device mlx5_0 \
    --size-bytes 67108864 \
    --batch-count 1

Client (machine B):
  python -m sglang.tools.mooncake_rdma_bw \
    --role client \
    --host 10.0.0.2 \
    --gpu-id 0 \
    --ib-device mlx5_0 \
    --size-bytes 67108864 \
    --duration 10 \
    --session-id "[10.0.0.1]:<rpc_port>" \
    --peer-addr <printed_by_server> \
    --batch-count 1

Batch mode (N buffers, size-bytes per buffer). Server prints comma-separated peer_addrs:
  # Server
  python -m sglang.tools.mooncake_rdma_bw \
    --role server \
    --host 10.0.0.1 \
    --gpu-id 0 \
    --ib-device mlx5_0 \
    --device cuda \
    --size-bytes 33554432 \
    --batch-count 4

  # Client (use --peer-addrs with the server-printed list)
  python -m sglang.tools.mooncake_rdma_bw \
    --role client \
    --host 10.0.0.2 \
    --gpu-id 0 \
    --ib-device mlx5_0 \
    --device cuda \
    --size-bytes 33554432 \
    --duration 10 \
    --session-id "[10.0.0.1]:<rpc_port>" \
    --peer-addrs "<addr1,addr2,addr3,addr4>" \
    --batch-count 4

Notes:
- `--host` should be the local IP/hostname of each machine that peers can reach.
- This script supports allocating buffers on `cpu` (host memory) or `cuda` via `--device`.
- For CUDA, requires PyTorch; for CPU, uses ctypes.

Optional ZMQ-based handshake (PUSH/PULL, single shot):
- Server: add `--zmq-push-bind tcp://*:5555` (send once and exit or continue)
- Client: add `--zmq-pull-connect tcp://10.0.0.1:5555` (recv once, then run)
  This shares `session_id`, `peer_addrs`, `size_bytes`, `batch_count` automatically.
"""

import argparse
import ctypes
import math
import os
import sys
import time
from typing import Optional, Tuple, List
import json

try:
    import torch  # Optional, only needed if --device cuda
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import zmq  # Optional, only used when ZMQ options provided
except Exception:  # pragma: no cover
    zmq = None  # type: ignore

from sglang.srt.disaggregation.mooncake.transfer_engine import (
    MooncakeTransferEngine,
)


def allocate_cpu_buffer(size_bytes: int) -> Tuple[int, ctypes.Array]:
    """Allocate a pinned python buffer (raw) and return (ptr, buf)."""
    buf = ctypes.create_string_buffer(size_bytes)
    ptr = ctypes.addressof(buf)
    return ptr, buf


def allocate_cuda_buffer(size_bytes: int) -> Tuple[int, "torch.Tensor"]:  # type: ignore[name-defined]
    if torch is None:
        raise RuntimeError("PyTorch is required for --device cuda")
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
    size_bytes: int,
    device: str,
    batch_count: int,
    zmq_push_bind: Optional[str],
):
    engine = MooncakeTransferEngine(hostname=host, gpu_id=gpu_id, ib_device=ib_device)

    ptrs: List[int] = []
    holders: List[object] = []
    for _ in range(max(1, batch_count)):
        if device == "cuda":
            ptr, holder = allocate_cuda_buffer(size_bytes)
        else:
            ptr, holder = allocate_cpu_buffer(size_bytes)
        engine.register(ptr, size_bytes)
        ptrs.append(ptr)
        holders.append(holder)

    print("Mooncake RDMA Server ready")
    print(f"session_id={engine.session_id}")
    if batch_count <= 1:
        print(f"peer_addr={ptrs[0]}")
    else:
        print("peer_addrs=" + ",".join(str(x) for x in ptrs))
    print(f"size_bytes={size_bytes/1024} kb")
    print(f"blocks={batch_count}")

    # Optional ZMQ one-shot metadata push (PUSH)
    if zmq_push_bind is not None:
        if zmq is None:
            raise RuntimeError("pyzmq is required when --zmq-push-bind is specified")
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUSH)
        socket.bind(zmq_push_bind)
        md = {
            "session_id": engine.session_id,
            "peer_addrs": ptrs,
            "size_bytes": size_bytes,
            "batch_count": batch_count,
        }
        socket.send_string(json.dumps(md))
        print(f"ZMQ PUSH sent metadata to {zmq_push_bind}")

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
    size_bytes: int,
    duration_sec: float,
    session_id: str,
    peer_addr: Optional[int],
    peer_addrs: Optional[List[int]],
    device: str,
    warmup_iters: int,
    report_interval_sec: float,
    batch_count: int,
    zmq_pull_connect: Optional[str],
):
    # Optional ZMQ one-shot metadata pull (PULL)
    if zmq_pull_connect is not None:
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
        # Fill missing fields from metadata (do not override explicitly provided args)
        if not session_id and "session_id" in md:
            session_id = md["session_id"]
        if (peer_addrs is None or len(peer_addrs) == 0) and "peer_addrs" in md:
            if isinstance(md["peer_addrs"], list):
                peer_addrs = [int(x) for x in md["peer_addrs"]]
        if not peer_addr and peer_addrs and len(peer_addrs) == 1:
            peer_addr = peer_addrs[0]
        if size_bytes <= 0 and "size_bytes" in md:
            size_bytes = int(md["size_bytes"])
        if batch_count <= 1 and "batch_count" in md:
            batch_count = int(md["batch_count"]) 

    engine = MooncakeTransferEngine(hostname=host, gpu_id=gpu_id, ib_device=ib_device)

    num_bufs = max(1, batch_count)
    src_ptrs: List[int] = []
    holders: List[object] = []
    for _ in range(num_bufs):
        if device == "cuda":
            ptr, holder = allocate_cuda_buffer(size_bytes)
        else:
            ptr, holder = allocate_cpu_buffer(size_bytes)
        src_ptrs.append(ptr)
        holders.append(holder)

    # Initialize the buffer with a simple pattern to avoid lazy allocation issues
    if device == "cuda":
        assert torch is not None
        for h in holders:
            h.fill_(0x7F7F7F7F)
        torch.cuda.synchronize()
    else:
        for p in src_ptrs:
            ctypes.memset(p, 0x5A, size_bytes)

    for p in src_ptrs:
        engine.register(p, size_bytes)

    if batch_count <= 1:
        if peer_addr is None:
            raise ValueError("client single-buffer mode requires --peer-addr")
        dst_addrs = [peer_addr]
    else:
        if not peer_addrs or len(peer_addrs) != num_bufs:
            raise ValueError("client batch mode requires --peer-addrs with length == --batch-count")
        dst_addrs = peer_addrs

    # Warmup
    for _ in range(max(0, warmup_iters)):
        if batch_count <= 1:
            ret = engine.transfer_sync(session_id, src_ptrs[0], dst_addrs[0], size_bytes)
        else:
            lengths = [size_bytes] * num_bufs
            ret = engine.batch_transfer_sync(session_id, src_ptrs, dst_addrs, lengths)
        if ret < 0:
            raise RuntimeError("Warmup transfer failed")

    if device == "cuda":
        torch.cuda.synchronize()  # type: ignore

    # Timed loop
    total_bytes = 0
    t0 = time.perf_counter()
    last_report = t0
    while True:
        if batch_count <= 1:
            ret = engine.transfer_sync(session_id, src_ptrs[0], dst_addrs[0], size_bytes)
            if ret < 0:
                raise RuntimeError("transfer_sync failed")
            total_bytes += size_bytes
        else:
            lengths = [size_bytes] * num_bufs
            ret = engine.batch_transfer_sync(session_id, src_ptrs, dst_addrs, lengths)
            if ret < 0:
                raise RuntimeError("batch_transfer_sync failed")
            total_bytes += size_bytes * num_bufs

        now = time.perf_counter()
        if now - t0 >= duration_sec:
            break
        if report_interval_sec > 0 and (now - last_report) >= report_interval_sec:
            bps = total_bytes / (now - t0)
            print(f"[progress] {human_bandwidth(bps)} over {now - t0:.2f}s")
            last_report = now

    if device == "cuda":
        torch.cuda.synchronize()  # type: ignore

    t1 = time.perf_counter()
    elapsed = max(1e-9, t1 - t0)
    bps = total_bytes / elapsed

    print("Mooncake RDMA Client result")
    print(f"size_bytes={size_bytes/1024} kb")
    print(f"duration={elapsed:.3f}s")
    print(f"bytes_transferred={total_bytes}")
    print(f"bandwidth={human_bandwidth(bps)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mooncake RDMA bandwidth benchmark")
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--host", type=str, required=True, help="Local host/IP")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--ib-device", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--size-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--batch-count", type=int, default=1, help="Number of buffers per transfer call")
    parser.add_argument("--zmq-push-bind", type=str, default=None, help="ZMQ PUSH bind address, e.g. tcp://*:5555 (server only)")
    parser.add_argument("--zmq-pull-connect", type=str, default=None, help="ZMQ PULL connect address, e.g. tcp://10.0.0.1:5555 (client only)")

    # client-only
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--peer-addr", type=int, default=None)
    parser.add_argument("--peer-addrs", type=str, default=None, help="Comma-separated list of destination buffer addresses for batch mode")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--report-interval", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.role == "server":
        run_server(
            host=args.host,
            gpu_id=args.gpu_id,
            ib_device=args.ib_device,
            size_bytes=args.size_bytes,
            device=args.device,
            batch_count=args.batch_count,
            zmq_push_bind=args.zmq_push_bind,
        )
        return

    # client: allow missing --session-id when using --zmq-pull-connect (will fetch via ZMQ)
    if not args.session_id and not args.zmq_pull_connect:
        print("--session-id is required for client role (unless --zmq-pull-connect is provided)", file=sys.stderr)
        sys.exit(2)

    parsed_peer_addrs: Optional[List[int]] = None
    if args.peer_addrs is not None:
        parsed_peer_addrs = [int(x.strip()) for x in args.peer_addrs.split(",") if x.strip()]

    run_client(
        host=args.host,
        gpu_id=args.gpu_id,
        ib_device=args.ib_device,
        size_bytes=args.size_bytes,
        duration_sec=args.duration,
        session_id=args.session_id,
        peer_addr=int(args.peer_addr) if args.peer_addr is not None else None,
        peer_addrs=parsed_peer_addrs,
        device=args.device,
        warmup_iters=args.warmup_iters,
        report_interval_sec=args.report_interval,
        batch_count=args.batch_count,
        zmq_pull_connect=args.zmq_pull_connect,
    )


if __name__ == "__main__":
    main()


