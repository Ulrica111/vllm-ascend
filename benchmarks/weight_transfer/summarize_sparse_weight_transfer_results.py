# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize real-Qwen3 Sparse HCCL and Sparse NPU IPC benchmark results.

This tool reads the JSON files produced by ``run_sparse_hccl_qwen3.sh`` and
``run_sparse_npu_ipc_qwen3.sh``. ``wire_bytes`` is the logical tensor payload
sent by the benchmark: BF16 values plus int32 flat indices. It deliberately
does not claim to be physical HCCL packet bytes.
"""

import argparse
import json
from pathlib import Path
from typing import Any


MIB = 2**20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the ten Sparse HCCL/IPC JSON result files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Write <prefix>.json and <prefix>.md.",
    )
    return parser.parse_args()


def read_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    backend = payload.get("backend", "sparse_hccl")
    if backend not in {"sparse_hccl", "sparse_ipc"}:
        raise ValueError(f"{path}: unsupported backend {backend!r}")
    return {
        "source": path.name,
        "backend": backend,
        "requested_ratio": result.get("requested_ratio", payload.get("ratio")),
        "actual_ratio": result["actual_ratio"],
        "updated_elements": result["updated_elements"],
        "wire_bytes": result["wire_bytes"],
        "parameter_count": result["parameter_count"],
        "trainer_send_mean_ms": result["metrics"]["trainer_send_ms"]["mean_ms"],
        "e2e_update_mean_ms": result["metrics"]["e2e_update_ms"]["mean_ms"],
        "incremental_peak_allocated_bytes": result["memory_metrics"][
            "incremental_allocated_bytes"
        ]["mean_bytes"],
        "ipc_update_request_counts": payload.get("ipc_update_request_counts"),
        "completion_after_update": result["completion_after_update"],
    }


def format_ratio(value: float) -> str:
    return f"{value * 100:g}%"


def format_mib(value: float) -> str:
    return f"{value / MIB:.2f}"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Sparse weight-transfer evaluation summary",
        "",
        "`wire_bytes` is the benchmark's logical tensor payload: BF16 values "
        "(2 bytes/element) plus int32 flat indices (4 bytes/element). It does "
        "not include HTTP/control metadata or physical HCCL packet overhead.",
        "",
        "| Backend | Ratio | Updated elements | Values (MiB) | Indices (MiB) | "
        "Logical wire (MiB) | Send mean (ms) | E2E mean (ms) | Trainer peak (MiB) | "
        "IPC requests | Completion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values_bytes = row["updated_elements"] * 2
        index_bytes = row["updated_elements"] * 4
        ipc_requests = row["ipc_update_request_counts"]
        request_text = "-" if ipc_requests is None else "/".join(map(str, ipc_requests))
        lines.append(
            "| {backend} | {ratio} | {elements:,} | {values} | {indices} | "
            "{wire} | {send:.2f} | {e2e:.2f} | {peak} | {requests} | {completion!r} |".format(
                backend=row["backend"],
                ratio=format_ratio(row["actual_ratio"]),
                elements=row["updated_elements"],
                values=format_mib(values_bytes),
                indices=format_mib(index_bytes),
                wire=format_mib(row["wire_bytes"]),
                send=row["trainer_send_mean_ms"],
                e2e=row["e2e_update_mean_ms"],
                peak=format_mib(row["incremental_peak_allocated_bytes"]),
                requests=request_text,
                completion=row["completion_after_update"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*_sparse_*_qwen3_4b.json"))
    if not paths:
        raise FileNotFoundError(f"No sparse Qwen3 result JSON files in {args.input_dir}")
    rows = [read_result(path) for path in paths]
    rows.sort(key=lambda row: (row["backend"], row["actual_ratio"]))

    backend_counts = {backend: sum(row["backend"] == backend for row in rows)
                      for backend in {"sparse_hccl", "sparse_ipc"}}
    if backend_counts != {"sparse_hccl": 5, "sparse_ipc": 5}:
        raise ValueError(
            "Expected five Sparse HCCL and five Sparse IPC files, got "
            f"{backend_counts}."
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(rows), encoding="utf-8")
    print(render_markdown(rows), end="")
    print(f"Wrote summary JSON: {json_path}")
    print(f"Wrote summary Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
