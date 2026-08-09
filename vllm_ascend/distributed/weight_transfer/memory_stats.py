# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU memory measurements for weight transfer benchmarks."""

from typing import Any

import torch


def capture_memory_baseline(memory: Any = torch.npu) -> dict[str, int]:
    """Reset peak counters and return the current transfer baseline."""
    baseline = {
        "allocated_bytes": memory.memory_allocated(),
        "reserved_bytes": memory.memory_reserved(),
    }
    memory.reset_peak_memory_stats()
    return baseline


def collect_peak_memory_stats(
    memory: Any,
    baseline: dict[str, int],
) -> dict[str, int]:
    """Return total and incremental peak memory since ``baseline``."""
    peak_allocated = memory.max_memory_allocated()
    peak_reserved = memory.max_memory_reserved()
    return {
        "baseline_allocated_bytes": baseline["allocated_bytes"],
        "baseline_reserved_bytes": baseline["reserved_bytes"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_allocated_bytes": max(
            0, peak_allocated - baseline["allocated_bytes"]
        ),
        "incremental_reserved_bytes": max(
            0, peak_reserved - baseline["reserved_bytes"]
        ),
    }
