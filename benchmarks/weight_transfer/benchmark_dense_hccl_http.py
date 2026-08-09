# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a full checkpoint dense packed HCCL update over HTTP."""

import argparse
import json
import math
import statistics
import threading
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
    HCCLWeightTransferEngine,
)
from vllm_ascend.distributed.weight_transfer.memory_stats import (
    capture_memory_baseline,
    collect_peak_memory_stats,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_BASE_URL = "http://localhost:8000"
PACKED_BUFFER_SIZE_BYTES = 256 * 2**20


def post(base_url: str, endpoint: str, payload: dict | None = None) -> None:
    response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=600)
    response.raise_for_status()


def get_world_size(base_url: str) -> int:
    response = requests.get(f"{base_url}/get_world_size", timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]


def build_update_info(parameters: dict[str, torch.Tensor]) -> dict:
    """Describe a packed full-checkpoint update to the HCCL worker."""
    return {
        "names": list(parameters),
        "dtype_names": [str(tensor.dtype).split(".")[-1]
                        for tensor in parameters.values()],
        "shapes": [list(tensor.shape) for tensor in parameters.values()],
        "packed": True,
        "packed_buffer_size_bytes": PACKED_BUFFER_SIZE_BYTES,
    }


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.9),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def summarize_bytes(values: list[int]) -> dict[str, float]:
    return {
        "mean_bytes": statistics.mean(values),
        "median_bytes": statistics.median(values),
        "p90_bytes": percentile(values, 0.9),
        "min_bytes": min(values),
        "max_bytes": max(values),
    }


class ThreadResult:
    error: BaseException | None = None


def post_async(
    base_url: str,
    endpoint: str,
    payload: dict | None,
    result: ThreadResult,
) -> None:
    try:
        post(base_url, endpoint, payload)
    except BaseException as exc:
        result.error = exc


def run_update(
    base_url: str,
    parameters: dict[str, torch.Tensor],
    update_info: dict,
    trainer_args: HCCLTrainerSendWeightsArgs,
) -> dict[str, object]:
    memory_baseline = capture_memory_baseline()
    start_e2e = time.perf_counter()
    start = time.perf_counter()
    post(base_url, "pause")
    pause_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    post(base_url, "start_weight_update")
    start_ms = (time.perf_counter() - start) * 1000

    result = ThreadResult()
    start = time.perf_counter()
    update_thread = threading.Thread(
        target=post_async,
        args=(base_url, "update_weights", {"update_info": update_info}, result),
    )
    update_thread.start()
    HCCLWeightTransferEngine.trainer_send_weights(
        iter(parameters.items()), trainer_args
    )
    torch.npu.synchronize()
    trainer_send_ms = (time.perf_counter() - start) * 1000
    update_thread.join()
    if result.error is not None:
        raise result.error
    server_update_rpc_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    post(base_url, "finish_weight_update")
    finish_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    post(base_url, "resume")
    resume_ms = (time.perf_counter() - start) * 1000
    return {
        "pause_ms": pause_ms,
        "start_ms": start_ms,
        "trainer_send_ms": trainer_send_ms,
        "server_update_rpc_ms": server_update_rpc_ms,
        "finish_ms": finish_ms,
        "resume_ms": resume_ms,
        "e2e_update_ms": (time.perf_counter() - start_e2e) * 1000,
        "memory": collect_peak_memory_stats(torch.npu, memory_baseline),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = get_world_size(args.base_url)
    if world_size != 1:
        raise RuntimeError("Dense packed benchmark currently requires server TP=1")
    device = f"npu:{world_size}"
    torch.accelerator.set_device_index(device)
    print(f"Loading trainer model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    parameters = {name: parameter.data for name, parameter in model.named_parameters()}
    update_info = build_update_info(parameters)
    checkpoint_bytes = sum(tensor.numel() * tensor.element_size()
                           for tensor in parameters.values())

    master_address, master_port = get_ip(), get_open_port()
    init_result = ThreadResult()
    init_thread = threading.Thread(
        target=post_async,
        args=(
            args.base_url,
            "init_weight_transfer_engine",
            {"init_info": {"master_address": master_address,
                            "master_port": master_port,
                            "rank_offset": 1,
                            "world_size": world_size + 1}},
            init_result,
        ),
    )
    init_thread.start()
    group = HCCLWeightTransferEngine.trainer_init(
        {"master_address": master_address, "master_port": master_port,
         "world_size": world_size + 1}
    )
    init_thread.join()
    if init_result.error is not None:
        raise init_result.error

    trainer_args = HCCLTrainerSendWeightsArgs(
        group=group,
        packed=True,
        packed_buffer_size_bytes=PACKED_BUFFER_SIZE_BYTES,
    )
    print(f"checkpoint={checkpoint_bytes / 2**30:.2f} GiB, packed_buffer=256 MiB")
    for _ in range(args.warmup):
        run_update(args.base_url, parameters, update_info, trainer_args)
    samples = [run_update(args.base_url, parameters, update_info, trainer_args)
               for _ in range(args.repeats)]
    metrics = {key: summarize([sample[key] for sample in samples])
               for key in samples[0] if key != "memory"}
    memory_metrics = {
        key: summarize_bytes([sample["memory"][key] for sample in samples])
        for key in samples[0]["memory"]
    }
    results = {
        "model": args.model,
        "update_scope": "full checkpoint (100%)",
        "packed_buffer_bytes": PACKED_BUFFER_SIZE_BYTES,
        "checkpoint_bytes": checkpoint_bytes,
        "parameter_count": len(parameters),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "metrics": metrics,
        "memory_metrics": memory_metrics,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results: {args.output}")


if __name__ == "__main__":
    main()
