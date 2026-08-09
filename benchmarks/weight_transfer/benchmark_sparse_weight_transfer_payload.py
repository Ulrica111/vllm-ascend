# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure fixed-payload Dense and Sparse HCCL/NPU-IPC data-plane updates.

The benchmark uses an 8 MiB BF16 runtime parameter by default. Dense sends a
complete tensor with zeroes outside the selected update positions; Sparse sends
the same selected positions as paired int32 indices and BF16 values. The final
receiver parameter is compared exactly so every row reports ``max_diff=0`` only
when both wire formats produce the same result.

This is a backend data-plane microbenchmark. Run the Qwen HTTP examples
separately to validate the full vLLM lifecycle.
"""

import argparse
import hashlib
import json
import multiprocessing
import queue
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
import torch_npu  # noqa: F401  # Register NPU multiprocessing reductions.
from torch.multiprocessing.reductions import reduce_tensor
from vllm.utils.network_utils import get_open_port

try:
    from benchmarks.weight_transfer.sparse_weight_transfer_metrics import (
        SparseTransferMeasurementRow,
        build_measurement_rows,
    )
except ModuleNotFoundError:
    # Direct ``python benchmarks/weight_transfer/<script>.py`` execution only
    # adds this directory, rather than the repository root, to ``sys.path``.
    from sparse_weight_transfer_metrics import (  # type: ignore[no-redef]
        SparseTransferMeasurementRow,
        build_measurement_rows,
    )
from vllm_ascend.distributed.weight_transfer.hccl_common import (
    stateless_init_process_group,
)
from vllm_ascend.distributed.weight_transfer.sparse_common import (
    SparseWeightPatch,
    apply_sparse_patch,
)


Backend = Literal["hccl", "ipc"]
Mode = Literal["dense", "sparse"]
DEFAULT_DENSE_PAYLOAD_BYTES = 8 * 2**20
DEFAULT_RATIOS = (0.001, 0.01, 0.1, 0.5, 1.0)
VALUE_DTYPE = torch.bfloat16
INDEX_DTYPE = torch.int32


class _OneParameterModel(torch.nn.Module):
    """Minimal runtime model used by the production sparse patch helper."""

    def __init__(self, elements: int, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(elements, dtype=VALUE_DTYPE, device=device),
            requires_grad=False,
        )


def _synchronize() -> None:
    torch.npu.synchronize()


def _reset_peak_memory() -> int:
    _synchronize()
    torch.npu.reset_peak_memory_stats()
    return torch.npu.memory_allocated()


def _peak_memory_delta(baseline: int) -> int:
    _synchronize()
    return max(0, torch.npu.max_memory_allocated() - baseline)


def _digest(tensor: torch.Tensor) -> str:
    """Return a stable digest after synchronizing device work."""
    _synchronize()
    # NumPy has no native BF16 scalar type in this environment. Every BF16
    # value is exactly representable as FP32, so this conversion preserves the
    # equality check while allowing a portable host-side digest.
    return hashlib.sha256(
        tensor.detach().float().cpu().numpy().tobytes()
    ).hexdigest()


def _build_source_tensors(
    *,
    elements: int,
    update_elements: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Dense and Sparse representations of the identical update."""
    stride = max(1, elements // update_elements)
    indices = (
        torch.arange(update_elements, device=device, dtype=INDEX_DTYPE) * stride
    )
    values = (
        torch.arange(update_elements, device=device, dtype=torch.float32)
        .remainder(97)
        .to(dtype=VALUE_DTYPE)
    )
    dense = torch.zeros(elements, dtype=VALUE_DTYPE, device=device)
    dense.index_copy_(0, indices.to(dtype=torch.long), values)
    return dense, indices, values


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual.float() - expected.float()).abs().max().detach().cpu().item()
    )


def _apply_dense(model: _OneParameterModel, received: torch.Tensor) -> None:
    model.weight.data.copy_(received)


def _apply_sparse(
    model: _OneParameterModel,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    apply_sparse_patch(
        model,
        SparseWeightPatch("weight", indices, values),
        expected_shape=list(model.weight.shape),
    )


def _hccl_worker(
    rank: int,
    device_index: int,
    port: int,
    mode: Mode,
    elements: int,
    update_elements: int,
    warmup: int,
    repeats: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Execute one HCCL mode; rank zero sends and rank one applies updates."""
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    group = stateless_init_process_group("127.0.0.1", port, rank, 2, device_index)
    model = _OneParameterModel(elements, device)
    dense, indices, values = _build_source_tensors(
        elements=elements,
        update_elements=update_elements,
        device=device,
    )
    expected = dense.detach().clone()
    sender_samples: list[float] = []
    receiver_peaks: list[int] = []

    for iteration in range(warmup + repeats):
        model.weight.data.zero_()
        baseline = _reset_peak_memory()
        start = time.perf_counter()
        if mode == "dense":
            received = dense if rank == 0 else torch.empty_like(dense)
            group.broadcast(received, src=0, stream=torch.npu.current_stream())
            if rank == 1:
                _apply_dense(model, received)
                del received
        else:
            received_indices = indices if rank == 0 else torch.empty_like(indices)
            received_values = values if rank == 0 else torch.empty_like(values)
            group.broadcast(
                received_indices, src=0, stream=torch.npu.current_stream()
            )
            group.broadcast(
                received_values, src=0, stream=torch.npu.current_stream()
            )
            if rank == 1:
                _apply_sparse(model, received_indices, received_values)
                del received_indices
                del received_values
        _synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        peak_bytes = _peak_memory_delta(baseline)
        if iteration >= warmup:
            if rank == 0:
                sender_samples.append(elapsed_ms)
            else:
                receiver_peaks.append(peak_bytes)

    if rank == 0:
        result_queue.put({"sender_ms": sender_samples})
    else:
        result_queue.put(
            {
                "receiver_peak_bytes": receiver_peaks,
                "digest": _digest(model.weight.data),
                "max_diff": _max_diff(model.weight.data, expected),
            }
        )


def _run_hccl_mode(
    *,
    mode: Mode,
    row: SparseTransferMeasurementRow,
    warmup: int,
    repeats: int,
    devices: tuple[int, int],
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    port = get_open_port()
    workers = [
        context.Process(
            target=_hccl_worker,
            args=(
                rank,
                devices[rank],
                port,
                mode,
                row.dense_payload_bytes // VALUE_DTYPE.itemsize,
                row.update_elements,
                warmup,
                repeats,
                result_queue,
            ),
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(600)
    timed_out = [worker.pid for worker in workers if worker.is_alive()]
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join()
    if timed_out:
        raise RuntimeError(f"HCCL workers timed out: {timed_out}")
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError(
            f"HCCL worker failures: {[worker.exitcode for worker in workers]}"
        )
    results = [_queue_get(result_queue, "HCCL") for _ in workers]
    return _merge_mode_results(results)


def _ipc_source(
    mode: Mode,
    device_index: int,
    elements: int,
    update_elements: int,
    warmup: int,
    repeats: int,
    payload_queue: multiprocessing.Queue,
    ack_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    """Create real Ascend IPC handles and keep their storage alive until ACK."""
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
        npu_generate_uuid,
    )

    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    dense, indices, values = _build_source_tensors(
        elements=elements,
        update_elements=update_elements,
        device=device,
    )
    npu_uuid = npu_generate_uuid(device_index)
    sender_samples: list[float] = []
    for iteration in range(warmup + repeats):
        start = time.perf_counter()
        if mode == "dense":
            _, dense_args = reduce_tensor(dense)
            payload_queue.put(("dense", npu_uuid, dense_args))
        else:
            _, index_args = reduce_tensor(indices)
            _, value_args = reduce_tensor(values)
            payload_queue.put(("sparse", npu_uuid, index_args, value_args))
        _queue_get(ack_queue, "IPC receiver acknowledgement")
        _synchronize()
        if iteration >= warmup:
            sender_samples.append((time.perf_counter() - start) * 1000)
    result_queue.put({"sender_ms": sender_samples})


def _rebuild_ipc_tensor(
    npu_uuid: str,
    rebuild_args: tuple,
    device_index: int,
) -> torch.Tensor:
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

    args = list(rebuild_args)
    args[6] = device_index
    return rebuild_npu_tensor(*args)


def _ipc_receiver(
    device_index: int,
    elements: int,
    update_elements: int,
    warmup: int,
    repeats: int,
    payload_queue: multiprocessing.Queue,
    ack_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    """Rebuild IPC tensors and exercise the same Dense/Sparse apply operations."""
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    model = _OneParameterModel(elements, device)
    _, expected_indices, expected_values = _build_source_tensors(
        elements=elements,
        update_elements=update_elements,
        device=device,
    )
    expected = torch.zeros(elements, dtype=VALUE_DTYPE, device=device)
    expected.index_copy_(0, expected_indices.to(dtype=torch.long), expected_values)
    receiver_peaks: list[int] = []
    for iteration in range(warmup + repeats):
        model.weight.data.zero_()
        baseline = _reset_peak_memory()
        payload = _queue_get(payload_queue, "IPC payload")
        mode = payload[0]
        if mode == "dense":
            _, npu_uuid, dense_args = payload
            received = _rebuild_ipc_tensor(npu_uuid, dense_args, device_index)
            _apply_dense(model, received)
        else:
            _, npu_uuid, index_args, value_args = payload
            received_indices = _rebuild_ipc_tensor(npu_uuid, index_args, device_index)
            received_values = _rebuild_ipc_tensor(npu_uuid, value_args, device_index)
            _apply_sparse(model, received_indices, received_values)
        peak_bytes = _peak_memory_delta(baseline)
        ack_queue.put(True)
        if iteration >= warmup:
            receiver_peaks.append(peak_bytes)

    result_queue.put(
        {
            "receiver_peak_bytes": receiver_peaks,
            "digest": _digest(model.weight.data),
            "max_diff": _max_diff(model.weight.data, expected),
        }
    )


def _run_ipc_mode(
    *,
    mode: Mode,
    row: SparseTransferMeasurementRow,
    warmup: int,
    repeats: int,
    device: int,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    payload_queue = context.Queue()
    ack_queue = context.Queue()
    result_queue = context.Queue()
    elements = row.dense_payload_bytes // VALUE_DTYPE.itemsize
    source = context.Process(
        target=_ipc_source,
        args=(
            mode,
            device,
            elements,
            row.update_elements,
            warmup,
            repeats,
            payload_queue,
            ack_queue,
            result_queue,
        ),
    )
    receiver = context.Process(
        target=_ipc_receiver,
        args=(
            device,
            elements,
            row.update_elements,
            warmup,
            repeats,
            payload_queue,
            ack_queue,
            result_queue,
        ),
    )
    source.start()
    receiver.start()
    source.join(600)
    receiver.join(600)
    workers = (source, receiver)
    timed_out = [worker.pid for worker in workers if worker.is_alive()]
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join()
    if timed_out:
        raise RuntimeError(f"IPC workers timed out: {timed_out}")
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError(
            f"IPC worker failures: {[worker.exitcode for worker in workers]}"
        )
    results = [_queue_get(result_queue, "IPC") for _ in workers]
    return _merge_mode_results(results)


def _queue_get(result_queue: multiprocessing.Queue, label: str) -> Any:
    try:
        return result_queue.get(timeout=30)
    except queue.Empty as exc:
        raise RuntimeError(f"{label} returned no result") from exc


def _merge_mode_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for result in results:
        merged.update(result)
    required = {"sender_ms", "receiver_peak_bytes", "digest", "max_diff"}
    missing = required.difference(merged)
    if missing:
        raise RuntimeError(f"Benchmark worker result missing {sorted(missing)}")
    return merged


def _median(values: list[float] | list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _run_row(
    backend: Backend,
    row: SparseTransferMeasurementRow,
    warmup: int,
    repeats: int,
    hccl_devices: tuple[int, int],
    ipc_device: int,
) -> dict[str, Any]:
    run_mode = _run_hccl_mode if backend == "hccl" else _run_ipc_mode
    if backend == "hccl":
        kwargs = {"devices": hccl_devices}
    else:
        kwargs = {"device": ipc_device}
    dense = run_mode(mode="dense", row=row, warmup=warmup, repeats=repeats, **kwargs)
    sparse = run_mode(mode="sparse", row=row, warmup=warmup, repeats=repeats, **kwargs)
    if dense["digest"] != sparse["digest"]:
        raise RuntimeError("Dense and Sparse receiver tensors have different digests")
    max_diff = max(float(dense["max_diff"]), float(sparse["max_diff"]))
    if max_diff != 0:
        raise RuntimeError(
            f"Dense/Sparse update correctness failed: max_diff={max_diff}"
        )
    return {
        **asdict(row),
        "savings_percent": row.savings_percent,
        "dense_median_ms": _median(dense["sender_ms"]),
        "sparse_median_ms": _median(sparse["sender_ms"]),
        "dense_peak_memory_bytes": int(_median(dense["receiver_peak_bytes"])),
        "sparse_peak_memory_bytes": int(_median(sparse["receiver_peak_bytes"])),
        "max_diff": max_diff,
        "dense_samples_ms": dense["sender_ms"],
        "sparse_samples_ms": sparse["sender_ms"],
    }


def _format_bytes(byte_count: int) -> str:
    if byte_count < 2**20:
        return f"{byte_count / 2**10:.2f} KiB"
    return f"{byte_count / 2**20:.2f} MiB"


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print(
        "ratio  dense/sparse payload      saving    dense ms  sparse ms  "
        "dense/sparse peak MiB  max_diff"
    )
    for row in rows:
        print(
            f"{row['ratio']:.1%}  "
            f"{_format_bytes(row['dense_payload_bytes']):>9} / "
            f"{_format_bytes(row['sparse_payload_bytes']):>9}  "
            f"{row['savings_percent']:7.2f}%  "
            f"{row['dense_median_ms']:8.3f}  {row['sparse_median_ms']:9.3f}  "
            f"{row['dense_peak_memory_bytes'] / 2**20:6.3f} / "
            f"{row['sparse_peak_memory_bytes'] / 2**20:6.3f}  "
            f"{row['max_diff']:.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hccl", "ipc"), required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--dense-payload-mib", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--hccl-devices", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--ipc-device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    rows = build_measurement_rows(
        dense_payload_bytes=args.dense_payload_mib * 2**20,
        ratios=args.ratios,
        value_bytes=VALUE_DTYPE.itemsize,
        index_bytes=INDEX_DTYPE.itemsize,
    )
    output_rows = [
        _run_row(
            args.backend,
            row,
            args.warmup,
            args.repeats,
            tuple(args.hccl_devices),
            args.ipc_device,
        )
        for row in rows
    ]
    _print_rows(output_rows)
    result = {
        "backend": args.backend,
        "method": (
            "fixed BF16 parameter; dense full tensor vs sparse int32 indices "
            "+ BF16 values"
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote results: {args.output}")


if __name__ == "__main__":
    main()
