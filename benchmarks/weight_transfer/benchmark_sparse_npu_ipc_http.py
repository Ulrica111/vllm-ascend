# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure verified real-Qwen3 Sparse NPU IPC updates through HTTP.

The trainer and vLLM server must use the same physical NPU. The trainer loads
the HF checkpoint, converts its Qwen3 parameters into vLLM's TP=1 runtime
layout, and sends flat ``indices`` plus ``values`` through the Sparse NPU IPC
backend. Start a server with ``{"backend":"sparse_ipc"}`` separately.
"""

import argparse
import json
import os
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM

from qwen3_runtime_weights import build_qwen3_runtime_parameters
from vllm_ascend.distributed.weight_transfer.memory_stats import (
    capture_memory_baseline,
    collect_peak_memory_stats,
)
from vllm_ascend.distributed.weight_transfer.sparse_common import (
    SparseWeightPatch,
)
from vllm_ascend.distributed.weight_transfer.sparse_npu_ipc_engine import (
    SparseNPUIPCTrainerSendWeightsArgs,
    SparseNPUIPCWeightTransferEngine,
)
from vllm_ascend.utils import vllm_version_is


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_VERIFY_PROMPT = "The future of AI is"


def plan_sparse_update_counts(
    named_numels: list[tuple[str, int]], ratio: float
) -> list[tuple[str, int]]:
    """Allocate a flat-index update count for every non-empty parameter."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    return [
        (name, int(numel * ratio))
        for name, numel in named_numels
        if int(numel * ratio) > 0
    ]


def get_world_size(base_url: str) -> int:
    response = requests.get(f"{base_url}/get_world_size", timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]


def build_completion_payload(model: str, prompt: str) -> dict[str, object]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
    }


def verify_completion(base_url: str, model: str, prompt: str) -> str:
    response = requests.post(
        f"{base_url}/v1/completions",
        json=build_completion_payload(model, prompt),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def post(base_url: str, endpoint: str, payload: dict[str, object] | None = None) -> None:
    response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=300)
    response.raise_for_status()


def iter_patch_chunks(
    parameters: dict[str, torch.Tensor],
    plan: list[tuple[str, int]],
    max_updates_per_request: int,
) -> Iterator[SparseWeightPatch]:
    """Yield bounded sparse patches without materializing all indices at once."""
    for name, count in plan:
        flat = parameters[name].detach().view(-1)
        stride = 1 if count == flat.numel() else flat.numel() // count
        for start in range(0, count, max_updates_per_request):
            stop = min(start + max_updates_per_request, count)
            indices = torch.arange(
                start, stop, device=flat.device, dtype=torch.int32
            )
            if stride != 1:
                indices.mul_(stride)
                values = flat.index_select(
                    0, indices.to(dtype=torch.long)
                ).contiguous()
            else:
                values = flat.narrow(0, start, stop - start)
            yield SparseWeightPatch(name=name, indices=indices, values=values)


def iter_patch_batches(
    parameters: dict[str, torch.Tensor],
    plan: list[tuple[str, int]],
    max_updates_per_request: int,
) -> Iterator[list[SparseWeightPatch]]:
    """Group bounded patches so one IPC request stays within its index budget."""
    batch: list[SparseWeightPatch] = []
    batch_updates = 0
    for patch in iter_patch_chunks(parameters, plan, max_updates_per_request):
        if batch and batch_updates + patch.indices.numel() > max_updates_per_request:
            yield batch
            batch = []
            batch_updates = 0
        batch.append(patch)
        batch_updates += patch.indices.numel()
    if batch:
        yield batch


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
    parameters: dict[str, torch.Tensor],
    plan: list[tuple[str, int]],
    trainer_args: SparseNPUIPCTrainerSendWeightsArgs,
    max_updates_per_request: int,
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
    ipc_update_request_count = 0
    for patches in iter_patch_batches(
        parameters, plan, max_updates_per_request
    ):
        SparseNPUIPCWeightTransferEngine.trainer_send_weights(
            iter(patches), trainer_args
        )
        ipc_update_request_count += 1
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
        "ipc_update_request_count": ipc_update_request_count,
        "memory": collect_peak_memory_stats(torch.npu, memory_baseline),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-updates-per-request", type=int, default=16_000_000)
    parser.add_argument("--verify-prompt", default=DEFAULT_VERIFY_PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not vllm_version_is("0.26.0"):
        raise RuntimeError(
            "This evaluation baseline uses the vLLM 0.26 static Sparse NPU "
            "IPC trainer API. Use the matching vLLM Ascend image."
        )
    if get_world_size(args.base_url) != 1:
        raise RuntimeError("Sparse NPU IPC benchmark currently requires server TP=1")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if args.max_updates_per_request <= 0:
        raise ValueError("max-updates-per-request must be positive")

    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    device = f"npu:{args.device}"
    torch.accelerator.set_device_index(device)
    print(f"Loading trainer model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    parameters = build_qwen3_runtime_parameters(model)
    noncontiguous = [name for name, value in parameters.items() if not value.is_contiguous()]
    if noncontiguous:
        raise RuntimeError(
            "Qwen3 runtime parameters must be contiguous: "
            + ", ".join(noncontiguous[:3])
        )

    plan = plan_sparse_update_counts(
        [(name, value.numel()) for name, value in parameters.items()], args.ratio
    )
    update_elements = sum(count for _, count in plan)
    wire_bytes = sum(count * (parameters[name].element_size() + 4) for name, count in plan)
    trainer_args = SparseNPUIPCTrainerSendWeightsArgs(
        send_mode="http",
        url=args.base_url,
        parameter_shapes={name: list(value.shape) for name, value in parameters.items()},
    )
    post(args.base_url, "init_weight_transfer_engine", {"init_info": {}})
    print(f"ratio={args.ratio:.3%}, elements={update_elements}, wire={wire_bytes / 2**30:.2f} GiB")
    for _ in range(args.warmup):
        run_update(
            args.base_url,
            parameters,
            plan,
            trainer_args,
            args.max_updates_per_request,
        )
    samples = [
        run_update(
            args.base_url,
            parameters,
            plan,
            trainer_args,
            args.max_updates_per_request,
        )
        for _ in range(args.repeats)
    ]
    completion = verify_completion(args.base_url, args.model, args.verify_prompt)
    metrics = {
        key: summarize([sample[key] for sample in samples])
        for key in samples[0]
        if key not in {"memory", "ipc_update_request_count"}
    }
    memory_metrics = {
        key: summarize_bytes([sample["memory"][key] for sample in samples])
        for key in samples[0]["memory"]
    }
    result = {
        "model": args.model,
        "backend": "sparse_ipc",
        "ratio_scope": "all mapped Qwen3 runtime parameters",
        "ratio": args.ratio,
        "actual_ratio": update_elements / sum(value.numel() for value in parameters.values()),
        "updated_elements": update_elements,
        "wire_bytes": wire_bytes,
        "parameter_count": len(plan),
        "max_updates_per_request": args.max_updates_per_request,
        "ipc_update_request_counts": [
            sample["ipc_update_request_count"] for sample in samples
        ],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "completion_after_update": completion,
        "metrics": metrics,
        "memory_metrics": memory_metrics,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote results: {args.output}")


if __name__ == "__main__":
    main()
