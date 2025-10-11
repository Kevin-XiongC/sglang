import os
import re
import json
import requests
import time
import argparse
import concurrent.futures
import asyncio
import aiohttp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from tqdm.asyncio import tqdm
matplotlib.use('Agg')  # 使用非交互式后端

# 全局统计列表
e2e_list = []
input_list = []
output_list = []
tps_list = []
ttft_list = []  # Time to First Token
itl_list = []   # Inter-Token Latency

@dataclass
class RequestOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

async def async_send_request(request_body, vllm_url, idx=None, pbar=None):
    """ 异步向 vllm 服务发送请求, 统计e2e耗时, 并解析input/output token数 """
    global e2e_list, input_list, output_list, tps_list, ttft_list, itl_list
    
    output = RequestOutput()
    prefix = f"[请求{idx}]" if idx is not None else ""
    
    try:
        # 创建异步HTTP会话
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        async with aiohttp.ClientSession(timeout=timeout) as session:
            request_start_time = time.perf_counter()
            
            # 发送请求
            async with session.post(vllm_url, json=request_body) as response:
                # print(f"{prefix} 状态码: {response.status}")
                
                if response.status != 200:
                    output.error = f"HTTP {response.status}"
                    output.success = False
                    if pbar:
                        pbar.update(1)
                    return output
                
                # 处理流式响应
                generated_text = ""
                ttft = 0.0
                most_recent_timestamp = request_start_time
                itl_times = []
                
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    
                    chunk = chunk_bytes.decode("utf-8")
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]  # Remove "data: " prefix
                    
                    if chunk == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        
                        # 检查是否有内容
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    # 第一个token
                                    if ttft == 0.0:
                                        ttft = timestamp - request_start_time
                                        output.ttft = ttft
                                    else:
                                        # 后续token的间隔时间
                                        itl_times.append(timestamp - most_recent_timestamp)
                                    
                                    most_recent_timestamp = timestamp
                                    generated_text += content
                        
                        # # 检查usage信息
                        # if "usage" in data:
                        #     usage = data["usage"]
                        #     output.input_tokens = usage.get("prompt_tokens", 0)
                        #     output.output_tokens = usage.get("completion_tokens", 0)
                    
                    except json.JSONDecodeError:
                        continue
                
                # 计算最终指标
                output.latency = time.perf_counter() - request_start_time
                output.generated_text = generated_text
                output.itl = itl_times
                output.success = True
                
                # 如果没有从usage获取到token数，尝试估算
                if output.input_tokens == 0:
                    # 简单估算：假设平均每个字符0.5个token
                    output.input_tokens = int(len(str(request_body.get("messages", [{}])[0].get("content", ""))) * 0.5)
                if output.output_tokens == 0:
                    output.output_tokens = int(len(generated_text) * 0.5)
                
                # 计算TPS
                if output.output_tokens > 0 and output.latency > 0:
                    tps = output.output_tokens / output.latency
                else:
                    tps = 0
                
                # 统计到全局列表
                e2e_list.append(output.latency)
                input_list.append(output.input_tokens)
                output_list.append(output.output_tokens)
                tps_list.append(tps)
                ttft_list.append(output.ttft)
                itl_list.extend(output.itl)
                
                print(f"{prefix} E2E耗时: {output.latency:.4f} 秒")
                print(f"{prefix} TTFT: {output.ttft:.4f} 秒")
                print(f"{prefix} ITL数量: {len(output.itl)}")
                if output.itl:
                    print(f"{prefix} ITL平均: {np.mean(output.itl)*1000:.2f} ms")
                print(f"{prefix} input tokens: {output.input_tokens}")
                print(f"{prefix} output tokens: {output.output_tokens}")
                print(f"{prefix} TPS: {tps:.2f}")
    
    except Exception as e:
        output.success = False
        output.error = str(e)
        print(f"{prefix} 请求失败：{e}")
    
    if pbar:
        pbar.update(1)
    return output

def send_request(request_body, ignore: bool = False, idx=None, vllm_url=None):
    """ 同步版本的请求函数，保持向后兼容 """
    # 为了保持向后兼容，这里调用异步版本
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_send_request(request_body, vllm_url, idx))
        return result.generated_text if result.success else None
    finally:
        loop.close()

def parse_log_file(filename):
    """ 解析日志文件，使用正则表达式按行分隔 """
    parsed_requests = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                print(f"解析 line 错误（第{line_number}行）：{e}")
                break

            request_body_raw = record.get("request_body").encode('utf-8').decode("unicode_escape")
            if request_body_raw is None:
                print(f"缺少 request_body 字段（第{line_number}行）")
                break

            try:
                request_body = json.loads(request_body_raw) if isinstance(request_body_raw, str) else request_body_raw
            except json.JSONDecodeError as e:
                print(f"解析 request_body 错误（第{line_number}行）：{e}")
                break
            parsed_requests.append(request_body)

    return parsed_requests

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    total_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p50_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p50_itl_ms: float
    p90_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p50_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float

def _get_current_request_rate(
    ramp_up_strategy: Optional[Literal["linear", "exponential"]],
    ramp_up_start_rps: Optional[int],
    ramp_up_end_rps: Optional[int],
    request_index: int,
    total_requests: int,
    request_rate: float,
) -> float:
    """Calculate the current request rate based on ramp-up strategy."""
    if (ramp_up_strategy and ramp_up_start_rps is not None
            and ramp_up_end_rps is not None):
        progress = request_index / max(total_requests - 1, 1)
        if ramp_up_strategy == "linear":
            increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
            return ramp_up_start_rps + increase
        elif ramp_up_strategy == "exponential":
            ratio = ramp_up_end_rps / ramp_up_start_rps
            return ramp_up_start_rps * (ratio**progress)
        else:
            raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
    return request_rate

def calculate_metrics(outputs: List[RequestOutput], dur_s: float) -> BenchmarkMetrics:
    """计算综合性能指标"""
    completed = sum(1 for output in outputs if output.success)
    total_input = sum(output.input_tokens for output in outputs if output.success)
    total_output = sum(output.output_tokens for output in outputs if output.success)
    
    # 提取各种延迟数据
    e2e_latencies = [output.latency for output in outputs if output.success]
    ttfts = [output.ttft for output in outputs if output.success and output.ttft > 0]
    itls = []
    
    for output in outputs:
        if output.success:
            itls.extend(output.itl)
    
    # 计算吞吐量
    request_throughput = completed / dur_s if dur_s > 0 else 0
    input_throughput = total_input / dur_s if dur_s > 0 else 0
    output_throughput = total_output / dur_s if dur_s > 0 else 0
    total_throughput = (total_input + total_output) / dur_s if dur_s > 0 else 0
    
    # 计算并发度
    concurrency = sum(e2e_latencies) / dur_s if dur_s > 0 else 0
    
    return BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=request_throughput,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        total_throughput=total_throughput,
        mean_ttft_ms=np.mean(ttfts) * 1000 if ttfts else 0,
        median_ttft_ms=np.median(ttfts) * 1000 if ttfts else 0,
        std_ttft_ms=np.std(ttfts) * 1000 if ttfts else 0,
        p50_ttft_ms=np.percentile(ttfts, 50) * 1000 if ttfts else 0,
        p90_ttft_ms=np.percentile(ttfts, 90) * 1000 if ttfts else 0,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000 if ttfts else 0,
        mean_itl_ms=np.mean(itls) * 1000 if itls else 0,
        median_itl_ms=np.median(itls) * 1000 if itls else 0,
        std_itl_ms=np.std(itls) * 1000 if itls else 0,
        p50_itl_ms=np.percentile(itls, 50) * 1000 if itls else 0,
        p90_itl_ms=np.percentile(itls, 90) * 1000 if itls else 0,
        p95_itl_ms=np.percentile(itls, 95) * 1000 if itls else 0,
        p99_itl_ms=np.percentile(itls, 99) * 1000 if itls else 0,
        max_itl_ms=np.max(itls) * 1000 if itls else 0,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000 if e2e_latencies else 0,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000 if e2e_latencies else 0,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000 if e2e_latencies else 0,
        p50_e2e_latency_ms=np.percentile(e2e_latencies, 50) * 1000 if e2e_latencies else 0,
        p90_e2e_latency_ms=np.percentile(e2e_latencies, 90) * 1000 if e2e_latencies else 0,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000 if e2e_latencies else 0,
        concurrency=concurrency,
    )

async def get_request_generator(
    input_requests: List[Dict], 
    request_rate: float,
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
):
    """生成请求的异步生成器，支持请求速率控制和ramp-up策略"""
    total_requests = len(input_requests)
    
    if request_rate == float("inf") and ramp_up_strategy is None:
        # 如果请求速率是无穷大且没有ramp-up，立即发送所有请求
        for i, request in enumerate(input_requests):
            yield i, request
    else:
        # 使用泊松过程控制请求速率，支持ramp-up
        for i, request in enumerate(input_requests):
            yield i, request
            
            if i < len(input_requests) - 1:  # 最后一个请求不需要等待
                # 计算当前请求速率
                current_rate = _get_current_request_rate(
                    ramp_up_strategy,
                    ramp_up_start_rps,
                    ramp_up_end_rps,
                    i,
                    total_requests,
                    request_rate
                )
                
                if current_rate == float("inf"):
                    continue  # 如果当前速率为无穷大，不等待
                
                # 从指数分布中采样请求间隔
                interval = np.random.exponential(1.0 / current_rate)
                await asyncio.sleep(interval)

async def async_benchmark(
    requests_data: List[Dict], 
    vllm_url: str, 
    request_rate: float, 
    max_concurrency: Optional[int] = None, 
    disable_tqdm: bool = False,
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
):
    """异步基准测试主函数"""
    global e2e_list, input_list, output_list, tps_list, ttft_list, itl_list
    
    # 清空全局统计列表
    e2e_list.clear()
    input_list.clear()
    output_list.clear()
    tps_list.clear()
    ttft_list.clear()
    itl_list.clear()
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    
    async def limited_request_func(request_body, idx, pbar):
        if semaphore:
            async with semaphore:
                return await async_send_request(request_body, vllm_url, idx, pbar)
        else:
            return await async_send_request(request_body, vllm_url, idx, pbar)
    
    # 开始基准测试
    benchmark_start_time = time.perf_counter()
    tasks = []
    pbar = None if disable_tqdm else tqdm(total=len(requests_data), desc="发送请求")
    
    # 创建请求生成器
    request_generator = get_request_generator(
        requests_data, 
        request_rate,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps
    )
    
    # 发送所有请求
    async for idx, request_body in request_generator:
        task = asyncio.create_task(limited_request_func(request_body, idx + 1, pbar))
        tasks.append(task)
    
    # 等待所有请求完成
    outputs = await asyncio.gather(*tasks)
    
    if pbar:
        pbar.close()
    
    # 计算基准测试持续时间
    benchmark_duration = time.perf_counter() - benchmark_start_time
    
    # 计算指标
    metrics = calculate_metrics(outputs, benchmark_duration)
    
    return metrics, outputs, benchmark_duration

def plot_histograms():
    """ 绘制E2E、input、output、TPS、TPOT的直方图，按10%分桶 """
    metrics = {
        'E2E (秒)': e2e_list,
        'Input Tokens (prompt_tokens)': input_list,
        'Output Tokens (completion_tokens)': output_list,
        'TPS (Tokens Per Second)': tps_list,
        'TPOT (Time Per Output Token)': [1.0/tps if tps > 0 else 0 for tps in tps_list]
    }
    
    # 创建2x3的子图布局（最后一个位置空着）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('性能指标直方图分布（10%分桶）', fontsize=16)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 展平axes数组以便索引
    axes_flat = axes.flatten()
    
    plot_idx = 0
    for metric_name, data in metrics.items():
        if not data:  # 如果没有数据则跳过
            continue
        
        ax = axes_flat[plot_idx]
        
        # 绘制直方图，使用10个bins（10%分桶）
        n, bins, patches = ax.hist(data, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_title(metric_name)
        ax.set_xlabel('值')
        ax.set_ylabel('频次')
        ax.grid(True, alpha=0.3)
        
        # 在每个柱子上显示数值
        for i, (patch, count) in enumerate(zip(patches, n)):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width()/2, patch.get_height(),
                       f'{int(count)}', ha='center', va='bottom')
        
        plot_idx += 1
    
    # 隐藏未使用的子图
    for i in range(plot_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('performance_histograms.png', dpi=300, bbox_inches='tight')
    print("\n直方图已保存为 performance_histograms.png")
    
    # 显示统计信息
    print("\n==== 直方图统计信息 ====")
    for metric_name, data in metrics.items():
        if data:
            print(f"{metric_name}: {len(data)} 个数据点")

def print_metrics(metrics: BenchmarkMetrics, duration: float):
    """打印详细的性能指标"""
    print("\n{s:{c}^{n}}".format(s=" Benchmark Results ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total output tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input throughput (tok/s):", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total throughput (tok/s):", metrics.total_throughput))
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", metrics.median_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("P50 E2E Latency (ms):", metrics.p50_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("P90 E2E Latency (ms):", metrics.p90_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", metrics.p99_e2e_latency_ms))
    
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P50 TTFT (ms):", metrics.p50_ttft_ms))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", metrics.p90_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    
    print("{s:{c}^{n}}".format(s="Inter-Token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P50 ITL (ms):", metrics.p50_itl_ms))
    print("{:<40} {:<10.2f}".format("P90 ITL (ms):", metrics.p90_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", metrics.p95_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{:<40} {:<10.2f}".format("Max ITL (ms):", metrics.max_itl_ms))
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM service with JSON request logs")
    parser.add_argument('--batchsize', type=int, default=1, help='并发请求数 (已弃用，使用 --max-concurrency)')
    parser.add_argument('--max-concurrency', type=int, default=None, help='最大并发请求数')
    parser.add_argument('--request-num', type=int, default=None, help='要抓取的请求体数量')
    parser.add_argument('--logfile', type=str, required=True, help='日志文件路径')
    parser.add_argument('--url', type=str, required=True, help='vllm服务URL')
    parser.add_argument('--dry-run', action='store_true', help='只解析请求体，不发送请求')
    parser.add_argument('--request-rate', type=float, default=float("inf"), 
                       help='请求速率 (req/s)，默认为无穷大（立即发送所有请求）')
    parser.add_argument('--disable-tqdm', action='store_true', help='禁用进度条')
    parser.add_argument('--output-file', type=str, help='输出结果到JSON文件')
    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="The ramp-up strategy. This would be used to "
        "ramp up the request rate from initial RPS to final "
        "RPS rate (specified by --ramp-up-start-rps and "
        "--ramp-up-end-rps.) over the duration of the benchmark."
    )
    parser.add_argument(
        "--ramp-up-start-rps",
        type=int,
        default=None,
        help="The starting request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ramp-up-end-rps",
        type=int,
        default=None,
        help="The ending request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    args = parser.parse_args()

    # Validate ramp-up arguments
    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError(
                "When using ramp-up, do not specify --request-rate. "
                "The request rate will be controlled by ramp-up parameters. "
                "Please remove the --request-rate argument."
            )
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "When using --ramp-up-strategy, both --ramp-up-start-rps and "
                "--ramp-up-end-rps must be specified"
            )
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up start and end RPS must be non-negative")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be less than or equal to end RPS")
        if (args.ramp_up_strategy == "exponential"
                and args.ramp_up_start_rps == 0):
            raise ValueError(
                "For exponential ramp-up, the start RPS cannot be 0.")

    requests_data = parse_log_file(args.logfile)
    total_requests = len(requests_data)
    print(f"parse_log_file 解析出 {total_requests} 个请求体")
    if total_requests == 0:
        print("没有解析到任何请求体，请检查日志文件格式")
        return

    # 确定要处理的请求数量
    if args.request_num is not None:
        max_requests = min(total_requests, args.request_num)
    else:
        max_requests = total_requests

    # 限制请求数量
    requests_data = requests_data[:max_requests]
    
    # 处理并发参数
    max_concurrency = args.max_concurrency
    if max_concurrency is None and args.batchsize != 1:
        max_concurrency = args.batchsize
        print(f"警告: --batchsize 已弃用，使用 --max-concurrency={max_concurrency}")

    if args.dry_run:
        print(f"max_requests: {max_requests}")
        print(f"request_rate: {args.request_rate}")
        print(f"max_concurrency: {max_concurrency}")
        return

    print(f"开始基准测试:")
    print(f"  请求数量: {max_requests}")
    if args.ramp_up_strategy is not None:
        print(f"  Ramp-up策略: {args.ramp_up_strategy}")
        print(f"  起始RPS: {args.ramp_up_start_rps}")
        print(f"  结束RPS: {args.ramp_up_end_rps}")
    else:
        print(f"  请求速率: {args.request_rate} req/s")
    print(f"  最大并发: {max_concurrency}")
    print(f"  服务URL: {args.url}")

    # 运行异步基准测试
    try:
        metrics, outputs, duration = asyncio.run(
            async_benchmark(
                requests_data=requests_data,
                vllm_url=args.url,
                request_rate=args.request_rate,
                max_concurrency=max_concurrency,
                disable_tqdm=args.disable_tqdm,
                ramp_up_strategy=args.ramp_up_strategy,
                ramp_up_start_rps=args.ramp_up_start_rps,
                ramp_up_end_rps=args.ramp_up_end_rps
            )
        )
        
        # 打印详细指标
        print_metrics(metrics, duration)
        
        # 保存结果到文件
        if args.output_file:
            result = {
                "args": vars(args),
                "metrics": {
                    "completed": metrics.completed,
                    "total_input": metrics.total_input,
                    "total_output": metrics.total_output,
                    "request_throughput": metrics.request_throughput,
                    "input_throughput": metrics.input_throughput,
                    "output_throughput": metrics.output_throughput,
                    "total_throughput": metrics.total_throughput,
                    "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
                    "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
                    "p50_e2e_latency_ms": metrics.p50_e2e_latency_ms,
                    "p90_e2e_latency_ms": metrics.p90_e2e_latency_ms,
                    "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
                    "mean_ttft_ms": metrics.mean_ttft_ms,
                    "median_ttft_ms": metrics.median_ttft_ms,
                    "p50_ttft_ms": metrics.p50_ttft_ms,
                    "p90_ttft_ms": metrics.p90_ttft_ms,
                    "p99_ttft_ms": metrics.p99_ttft_ms,
                    "mean_itl_ms": metrics.mean_itl_ms,
                    "median_itl_ms": metrics.median_itl_ms,
                    "p50_itl_ms": metrics.p50_itl_ms,
                    "p90_itl_ms": metrics.p90_itl_ms,
                    "p95_itl_ms": metrics.p95_itl_ms,
                    "p99_itl_ms": metrics.p99_itl_ms,
                    "max_itl_ms": metrics.max_itl_ms,
                    "concurrency": metrics.concurrency,
                },
                "duration": duration,
                "outputs": [
                    {
                        "success": output.success,
                        "latency": output.latency,
                        "ttft": output.ttft,
                        "input_tokens": output.input_tokens,
                        "output_tokens": output.output_tokens,
                        "error": output.error
                    } for output in outputs
                ]
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n结果已保存到: {args.output_file}")
        
        # 绘制直方图
        # plot_histograms()
        
    except Exception as e:
        print(f"基准测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()