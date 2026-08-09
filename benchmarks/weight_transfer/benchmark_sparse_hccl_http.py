# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure verified Sparse HCCL updates for Qwen3 with a running vLLM server."""

import argparse
import json
import math
import statistics
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
    HCCLWeightTransferEngine,
)
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLWeightTransferEngine,
    SparseWeightPatch,
)
from vllm_ascend.distributed.weight_transfer.memory_stats import (
    capture_memory_baseline,
    collect_peak_memory_stats,
)
from qwen3_runtime_weights import build_qwen3_runtime_parameters


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_VERIFY_PROMPT = "The future of AI is"


def plan_sparse_update_counts(
    named_numels: list[tuple[str, int]],
    ratio: float,
) -> list[tuple[str, int]]:
    """Allocate a ratio of every parameter, skipping zero-sized patches."""
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
    """Build a minimal curl-equivalent request for the OpenAI API."""
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
    }


def verify_completion(base_url: str, model: str, prompt: str) -> str:
    """Verify that the resumed server accepts an inference request."""
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


@dataclass
class ThreadResult:
    error: BaseException | None = None


def post_update_async(
    base_url: str,
    update_info: dict[str, object],
    result: ThreadResult,
) -> None:
    try:
        post(base_url, "update_weights", {"update_info": update_info})
    except BaseException as exc:  # Propagate HTTP failures to the benchmark.
        result.error = exc


def make_patch(name: str, parameter: torch.Tensor, count: int) -> SparseWeightPatch:
    """Build a deterministic, evenly spaced patch for one parameter."""
    flat = parameter.detach().view(-1)
    if count == flat.numel():
        indices = torch.arange(count, device=flat.device, dtype=torch.int32)
        values = flat
    else:
        stride = flat.numel() // count
        indices = torch.arange(count, device=flat.device, dtype=torch.int32) * stride
        values = flat.index_select(0, indices.to(dtype=torch.long)).contiguous()
    return SparseWeightPatch(name=name, indices=indices, values=values)


def iter_patches(
    parameters: dict[str, torch.Tensor],
    plan: list[tuple[str, int]],
) -> Iterator[SparseWeightPatch]:
    for name, count in plan:
        yield make_patch(name, parameters[name], count)


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


def run_update(
    base_url: str,
    parameters: dict[str, torch.Tensor],
    plan: list[tuple[str, int]],
    update_info: dict[str, object],
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

    update_result = ThreadResult()
    start = time.perf_counter()
    update_thread = threading.Thread(
        target=post_update_async,
        args=(base_url, update_info, update_result),
    )
    update_thread.start()
    SparseHCCLWeightTransferEngine.trainer_send_weights(
        iter_patches(parameters, plan), trainer_args
    )
    torch.npu.synchronize()
    trainer_send_ms = (time.perf_counter() - start) * 1000

    update_thread.join()
    if update_result.error is not None:
        raise update_result.error
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
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Fraction of all mapped Qwen3 runtime parameter elements to update.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--verify-prompt", default=DEFAULT_VERIFY_PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_world_size = get_world_size(args.base_url)
    if inference_world_size != 1:
        raise RuntimeError("Sparse HCCL benchmark currently requires server TP=1")
    device = f"npu:{inference_world_size}"
    torch.accelerator.set_device_index(device)

    print(f"Loading trainer model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    parameters = build_qwen3_runtime_parameters(model)
    noncontiguous = [name for name, parameter in parameters.items()
                     if not parameter.is_contiguous()]
    if noncontiguous:
        raise RuntimeError(
            "Qwen3 runtime parameters must be contiguous: "
            f"{', '.join(noncontiguous[:3])}"
        )
    total_elements = sum(parameter.numel() for parameter in parameters.values())

    master_address, master_port = get_ip(), get_open_port()
    init_result = ThreadResult()
    init_thread = threading.Thread(
        target=lambda: post_update_init(
            args.base_url,
            master_address,
            master_port,
            inference_world_size + 1,
            init_result,
        )
    )
    init_thread.start()
    group = HCCLWeightTransferEngine.trainer_init(
        {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": inference_world_size + 1,
        }
    )
    init_thread.join()
    if init_result.error is not None:
        raise init_result.error

    results: dict[str, object] = {
        "model": args.model,
        "base_url": args.base_url,
        "ratio_scope": "all mapped Qwen3 runtime parameters",
        "mapped_runtime_parameter_elements": total_elements,
        "mapped_runtime_parameter_count": len(parameters),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "ratio": args.ratio,
    }
    trainer_args = HCCLTrainerSendWeightsArgs(group=group)
    plan = plan_sparse_update_counts(
        [(name, parameter.numel()) for name, parameter in parameters.items()], args.ratio
    )
    update_elements = sum(count for _, count in plan)
    update_info = {
        "names": [name for name, _ in plan],
        "dtype_names": [str(parameters[name].dtype).split(".")[-1] for name, _ in plan],
        "shapes": [list(parameters[name].shape) for name, _ in plan],
        "num_updates_list": [count for _, count in plan],
    }
    wire_bytes = sum(
        count * (torch.tensor([], dtype=parameters[name].dtype).element_size() + 4)
        for name, count in plan
    )
    print(
        f"ratio={args.ratio:.3%}, elements={update_elements}, "
        f"wire={wire_bytes / 2**30:.2f} GiB"
    )
    for _ in range(args.warmup):
        run_update(args.base_url, parameters, plan, update_info, trainer_args)
    samples = [
        run_update(args.base_url, parameters, plan, update_info, trainer_args)
        for _ in range(args.repeats)
    ]
    completion = verify_completion(args.base_url, args.model, args.verify_prompt)
    print(f"completion_after_update={completion!r}")
    metrics = {
        key: summarize([sample[key] for sample in samples])
        for key in samples[0]
        if key != "memory"
    }
    memory_metrics = {
        key: summarize_bytes([sample["memory"][key] for sample in samples])
        for key in samples[0]["memory"]
    }
    results["result"] = {
        "requested_ratio": args.ratio,
        "actual_ratio": update_elements / total_elements,
        "updated_elements": update_elements,
        "wire_bytes": wire_bytes,
        "wire_gib": wire_bytes / 2**30,
        "parameter_count": len(plan),
        "exact_server_validation": "passed when update_weights returned HTTP 200",
        "completion_after_update": completion,
        "metrics": metrics,
        "memory_metrics": memory_metrics,
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results: {args.output}")


def post_update_init(
    base_url: str,
    master_address: str,
    master_port: int,
    world_size: int,
    result: ThreadResult,
) -> None:
    try:
        post(
            base_url,
            "init_weight_transfer_engine",
            {
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": world_size,
                }
            },
        )
    except BaseException as exc:
        result.error = exc


if __name__ == "__main__":
    main()
