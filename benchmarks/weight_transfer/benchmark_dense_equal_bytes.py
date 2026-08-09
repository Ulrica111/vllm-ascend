# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure HCCL or NPU IPC Dense transport at Sparse-equivalent byte counts.

This is a transport microbenchmark, not a vLLM weight-update benchmark.  It
uses bytes copied from Qwen3-4B's TP=1 runtime parameters, but sends them in
standalone BF16 chunks.  Therefore the Dense payload can exactly match a
Sparse patch's logical wire size: BF16 values plus int32 indices.

HCCL measures a real two-NPU broadcast.  IPC measures real producer/consumer
NPU IPC handle mapping on one physical NPU; it must not be interpreted as
cross-card traffic.  Buffer preparation and exact validation are deliberately
outside the timed section.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import statistics
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from qwen3_runtime_weights import build_qwen3_runtime_parameters
from vllm_ascend.distributed.weight_transfer.hccl_common import (
    stateless_init_process_group,
)
from vllm_ascend.distributed.weight_transfer.memory_stats import (
    capture_memory_baseline,
    collect_peak_memory_stats,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_CHUNK_MIB = 256
SPARSE_INDEX_BYTES = 4


def plan_sparse_update_counts(
    named_numels: list[tuple[str, int]], ratio: float
) -> list[tuple[str, int]]:
    """Match the existing Sparse benchmark's per-runtime-parameter plan."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    return [
        (name, int(numel * ratio))
        for name, numel in named_numels
        if int(numel * ratio) > 0
    ]


def sparse_wire_bytes(
    parameters: dict[str, torch.Tensor], ratio: float
) -> tuple[int, int, float]:
    """Return Sparse logical bytes and exact update ratio for this model."""
    plan = plan_sparse_update_counts(
        [(name, value.numel()) for name, value in parameters.items()], ratio
    )
    update_elements = sum(count for _, count in plan)
    value_bytes = sum(
        count * parameters[name].element_size() for name, count in plan
    )
    return (
        value_bytes + update_elements * SPARSE_INDEX_BYTES,
        update_elements,
        update_elements / sum(value.numel() for value in parameters.values()),
    )


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


def digest_tensor(tensor: torch.Tensor) -> str:
    """Return an exact byte digest without relying on NumPy BF16 support."""
    raw = tensor.detach().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class RuntimeChunkSource:
    """Cycles through real Qwen runtime tensors to fill Dense BF16 chunks."""

    tensors: list[torch.Tensor]
    tensor_index: int = 0
    tensor_offset: int = 0

    def make(self, elements: int, device: str) -> tuple[torch.Tensor, float]:
        start = time.perf_counter()
        output = torch.empty(elements, dtype=torch.bfloat16, device=device)
        written = 0
        while written < elements:
            source = self.tensors[self.tensor_index].detach().view(-1)
            available = source.numel() - self.tensor_offset
            count = min(elements - written, available)
            output.narrow(0, written, count).copy_(
                source.narrow(0, self.tensor_offset, count)
            )
            written += count
            self.tensor_offset += count
            if self.tensor_offset == source.numel():
                self.tensor_index = (self.tensor_index + 1) % len(self.tensors)
                self.tensor_offset = 0
        torch.npu.synchronize()
        preparation_ms = (time.perf_counter() - start) * 1000
        return output, preparation_ms


def chunk_sizes(total_bytes: int, chunk_bytes: int) -> Iterator[int]:
    if total_bytes % torch.tensor([], dtype=torch.bfloat16).element_size():
        raise ValueError("Dense logical byte target must be BF16-aligned")
    while total_bytes:
        current = min(total_bytes, chunk_bytes)
        current -= current % 2
        yield current
        total_bytes -= current


def load_source(model_name: str, device: str) -> RuntimeChunkSource:
    print(f"Loading real Qwen trainer source: {model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).to(device)
    parameters = build_qwen3_runtime_parameters(model)
    return RuntimeChunkSource(list(parameters.values()))


def _put_error(results: mp.Queue, rank: int, exc: BaseException) -> None:
    results.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})


def hccl_worker(
    rank: int,
    model: str,
    total_bytes: int,
    chunk_bytes: int,
    warmup: int,
    repeats: int,
    master_address: str,
    master_port: int,
    results: mp.Queue,
) -> None:
    """Run the HCCL producer or receiver for equal-byte Dense transport."""
    try:
        device = f"npu:{rank}"
        torch.accelerator.set_device_index(device)
        group = stateless_init_process_group(
            master_address, master_port, rank, 2, rank
        )
        source = load_source(model, device) if rank == 0 else None
        sizes = list(chunk_sizes(total_bytes, chunk_bytes))

        def transfer(timed: bool) -> tuple[float, float, list[str]]:
            transfer_ms = 0.0
            preparation_ms = 0.0
            digests: list[str] = []
            for size in sizes:
                if rank == 0:
                    assert source is not None
                    tensor, prepared = source.make(size // 2, device)
                    preparation_ms += prepared
                else:
                    tensor = torch.empty(size // 2, dtype=torch.bfloat16, device=device)
                torch.npu.synchronize()
                start = time.perf_counter()
                group.broadcast(tensor, src=0)
                torch.npu.synchronize()
                transfer_ms += (time.perf_counter() - start) * 1000
                if not timed:
                    digests.append(digest_tensor(tensor))
                del tensor
            return transfer_ms, preparation_ms, digests

        for _ in range(warmup):
            transfer(timed=True)
        samples = []
        for _ in range(repeats):
            memory_baseline = capture_memory_baseline()
            transfer_ms, preparation_ms, _ = transfer(timed=True)
            samples.append({
                "transfer_ms": transfer_ms,
                "preparation_ms": preparation_ms,
                "memory": collect_peak_memory_stats(torch.npu, memory_baseline),
            })
        _, _, validation_digests = transfer(timed=False)
        results.put({
            "rank": rank,
            "samples": samples,
            "validation_digests": validation_digests,
        })
    except BaseException as exc:  # Surface child errors in the parent process.
        _put_error(results, rank, exc)


def ipc_worker(
    role: str,
    model: str,
    total_bytes: int,
    chunk_bytes: int,
    warmup: int,
    repeats: int,
    producer_to_consumer: mp.Queue,
    consumer_to_producer: mp.Queue,
    results: mp.Queue,
) -> None:
    """Measure real same-NPU IPC handle mapping using independently allocated chunks."""
    try:
        import torch_npu  # noqa: F401  # Registers NPU IPC reductions.
        from torch.multiprocessing.reductions import reduce_tensor
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        device = "npu:0"
        torch.accelerator.set_device_index(device)
        sizes = list(chunk_sizes(total_bytes, chunk_bytes))
        if role == "producer":
            source = load_source(model, device)
            def transfer(timed: bool) -> tuple[float, float, list[str]]:
                transfer_ms = 0.0
                preparation_ms = 0.0
                digests: list[str] = []
                for index, size in enumerate(sizes):
                    tensor, prepared = source.make(size // 2, device)
                    preparation_ms += prepared
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    _, ipc_args = reduce_tensor(tensor)
                    producer_to_consumer.put(
                        {
                            "kind": "chunk",
                            "index": index,
                            "args": ipc_args,
                            "validate": not timed,
                        }
                    )
                    acknowledgement = consumer_to_producer.get(timeout=600)
                    if acknowledgement != {"kind": "ack", "index": index}:
                        raise RuntimeError(f"Unexpected IPC acknowledgement: {acknowledgement}")
                    torch.npu.synchronize()
                    transfer_ms += (time.perf_counter() - start) * 1000
                    if not timed:
                        digests.append(digest_tensor(tensor))
                    del tensor
                return transfer_ms, preparation_ms, digests

            for _ in range(warmup):
                transfer(timed=True)
            samples = []
            for _ in range(repeats):
                memory_baseline = capture_memory_baseline()
                transfer_ms, preparation_ms, _ = transfer(timed=True)
                samples.append({
                    "transfer_ms": transfer_ms,
                    "preparation_ms": preparation_ms,
                    "memory": collect_peak_memory_stats(torch.npu, memory_baseline),
                })
            _, _, validation_digests = transfer(timed=False)
            producer_to_consumer.put({"kind": "stop"})
            results.put({
                "role": role,
                "samples": samples,
                "validation_digests": validation_digests,
            })
        else:
            validation_digests: list[str] = []
            while True:
                message = producer_to_consumer.get(timeout=600)
                if message["kind"] == "stop":
                    break
                if message["kind"] != "chunk":
                    raise RuntimeError(f"Unexpected IPC message: {message}")
                args = list(message["args"])
                args[6] = 0
                tensor = rebuild_npu_tensor(*args)
                torch.npu.synchronize()
                if message["validate"]:
                    validation_digests.append(digest_tensor(tensor))
                consumer_to_producer.put({"kind": "ack", "index": message["index"]})
                del tensor
            results.put({
                "role": role,
                "samples": [],
                "validation_digests": validation_digests,
            })
    except BaseException as exc:
        _put_error(results, -1, exc)


def collect_results(results: mp.Queue, expected: int) -> list[dict]:
    received = []
    while len(received) < expected:
        try:
            item = results.get(timeout=900)
        except queue.Empty as exc:
            raise RuntimeError("Timed out waiting for benchmark workers") from exc
        if "error" in item:
            raise RuntimeError(f"Benchmark worker failed: {item['error']}")
        received.append(item)
    return received


def run_hccl(args: argparse.Namespace, total_bytes: int, chunk_bytes: int) -> dict:
    context = mp.get_context("spawn")
    results = context.Queue()
    master_address, master_port = get_ip(), get_open_port()
    workers = [
        context.Process(
            target=hccl_worker,
            args=(rank, args.model, total_bytes, chunk_bytes, args.warmup,
                  args.repeats, master_address, master_port, results),
        )
        for rank in (0, 1)
    ]
    for worker in workers:
        worker.start()
    data = collect_results(results, 2)
    for worker in workers:
        worker.join(timeout=60)
        if worker.exitcode != 0:
            raise RuntimeError(f"HCCL worker exited with {worker.exitcode}")
    producer, receiver = sorted(data, key=lambda item: item["rank"])
    if producer["validation_digests"] != receiver["validation_digests"]:
        raise RuntimeError("HCCL exact byte validation failed")
    return producer


def run_ipc(args: argparse.Namespace, total_bytes: int, chunk_bytes: int) -> dict:
    context = mp.get_context("spawn")
    producer_to_consumer = context.Queue()
    consumer_to_producer = context.Queue()
    results = context.Queue()
    consumer = context.Process(
        target=ipc_worker,
        args=("consumer", args.model, total_bytes, chunk_bytes, args.warmup,
              args.repeats, producer_to_consumer, consumer_to_producer, results),
    )
    producer = context.Process(
        target=ipc_worker,
        args=("producer", args.model, total_bytes, chunk_bytes, args.warmup,
              args.repeats, producer_to_consumer, consumer_to_producer, results),
    )
    consumer.start()
    producer.start()
    data = collect_results(results, 2)
    for worker in (producer, consumer):
        worker.join(timeout=60)
        if worker.exitcode != 0:
            raise RuntimeError(f"IPC worker exited with {worker.exitcode}")
    producer_data = next(item for item in data if item.get("role") == "producer")
    consumer_data = next(item for item in data if item.get("role") == "consumer")
    if producer_data["validation_digests"] != consumer_data["validation_digests"]:
        raise RuntimeError("NPU IPC exact byte validation failed")
    return producer_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hccl", "ipc"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_mib <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("chunk-mib must be positive; warmup >= 0; repeats > 0")

    # Load only to derive the exact Qwen-runtime Sparse plan. Child producer
    # processes load the same real source used to populate timed Dense chunks.
    cpu_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    parameters = build_qwen3_runtime_parameters(cpu_model)
    total_bytes, update_elements, actual_ratio = sparse_wire_bytes(parameters, args.ratio)
    del cpu_model, parameters
    chunk_bytes = args.chunk_mib * 2**20
    print(
        f"backend={args.backend}, requested_ratio={args.ratio:.3%}, "
        f"sparse_equivalent_bytes={total_bytes}, chunks={len(list(chunk_sizes(total_bytes, chunk_bytes)))}",
        flush=True,
    )
    producer = (
        run_hccl(args, total_bytes, chunk_bytes)
        if args.backend == "hccl"
        else run_ipc(args, total_bytes, chunk_bytes)
    )
    samples = producer["samples"]
    result = {
        "model": args.model,
        "backend": args.backend,
        "benchmark_scope": "Dense transport microbenchmark; not a vLLM weight update",
        "source": "real Qwen3 TP=1 runtime parameter values",
        "requested_sparse_ratio": args.ratio,
        "actual_sparse_ratio": actual_ratio,
        "sparse_updated_elements": update_elements,
        "dense_payload_bytes": total_bytes,
        "dense_payload_mib": total_bytes / 2**20,
        "payload_definition": "exactly equals Sparse BF16 values plus int32 indices logical bytes",
        "chunk_bytes": chunk_bytes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "verification": "passed: producer and receiver SHA-256 chunk digests match; outside timed section",
        "metrics": {
            "transfer_ms": summarize([sample["transfer_ms"] for sample in samples]),
            "preparation_ms": summarize([sample["preparation_ms"] for sample in samples]),
        },
        "memory_metrics": {
            key: summarize_bytes([sample["memory"][key] for sample in samples])
            for key in samples[0]["memory"]
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote results: {args.output}")


if __name__ == "__main__":
    main()
