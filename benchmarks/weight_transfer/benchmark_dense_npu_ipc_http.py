# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a real-Qwen3 full Dense NPU IPC update through HTTP.

Start a vLLM server separately with ``{"backend":"ipc"}`` on the same
physical NPU as this process. This is the full-checkpoint baseline for the
Sparse NPU IPC five-ratio measurements.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM

from vllm_ascend.distributed.weight_transfer.memory_stats import (
    capture_memory_baseline,
    collect_peak_memory_stats,
)
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerSendWeightsArgs,
    NPUIPCWeightTransferEngine,
)
from vllm_ascend.utils import vllm_version_is


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def post(base_url: str, endpoint: str, payload: dict | None = None) -> None:
    response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=600)
    response.raise_for_status()


def get_world_size(base_url: str) -> int:
    response = requests.get(f"{base_url}/get_world_size", timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int((len(values) - 1) * fraction)]


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
        "p90_bytes": percentile([float(value) for value in values], 0.9),
        "min_bytes": min(values),
        "max_bytes": max(values),
    }


def run_update(
    base_url: str,
    model: torch.nn.Module,
    trainer_args: NPUIPCTrainerSendWeightsArgs,
) -> dict[str, object]:
    memory_baseline = capture_memory_baseline()
    start_e2e = time.perf_counter()
    start = time.perf_counter()
    post(base_url, "pause")
    pause_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    post(base_url, "start_weight_update")
    start_ms = (time.perf_counter() - start) * 1000

    torch.npu.synchronize()
    start = time.perf_counter()
    NPUIPCWeightTransferEngine.trainer_send_weights(
        model.named_parameters(), trainer_args
    )
    torch.npu.synchronize()
    trainer_send_ms = (time.perf_counter() - start) * 1000

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
        "finish_ms": finish_ms,
        "resume_ms": resume_ms,
        "e2e_update_ms": (time.perf_counter() - start_e2e) * 1000,
        "memory": collect_peak_memory_stats(torch.npu, memory_baseline),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not vllm_version_is("0.26.0"):
        raise RuntimeError(
            "This evaluation baseline uses the vLLM 0.26 static NPU IPC "
            "trainer API. Use the matching vLLM Ascend image."
        )
    if get_world_size(args.base_url) != 1:
        raise RuntimeError("Dense NPU IPC benchmark currently requires server TP=1")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    device = f"npu:{args.device}"
    torch.accelerator.set_device_index(device)
    print(f"Loading trainer model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    parameter_count = sum(1 for _ in model.named_parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode="http", url=args.base_url)
    post(args.base_url, "init_weight_transfer_engine", {"init_info": {}})
    for _ in range(args.warmup):
        run_update(args.base_url, model, trainer_args)
    samples = [
        run_update(args.base_url, model, trainer_args) for _ in range(args.repeats)
    ]
    metrics = {
        key: summarize([sample[key] for sample in samples])
        for key in samples[0]
        if key != "memory"
    }
    memory_metrics = {
        key: summarize_bytes([sample["memory"][key] for sample in samples])
        for key in samples[0]["memory"]
    }
    result = {
        "model": args.model,
        "backend": "ipc",
        "update_scope": "full checkpoint",
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "metrics": metrics,
        "memory_metrics": memory_metrics,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote results: {args.output}")


if __name__ == "__main__":
    main()
