# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral accounting for fixed-payload sparse transfer benchmarks."""

from dataclasses import dataclass
import math
from collections.abc import Iterable


@dataclass(frozen=True)
class SparseTransferMeasurementRow:
    """One requested update rate in the fixed Dense-payload experiment."""

    ratio: float
    dense_payload_bytes: int
    update_elements: int
    sparse_payload_bytes: int

    @property
    def savings_percent(self) -> float:
        """Logical payload saving relative to the complete Dense tensor."""
        return (1 - self.sparse_payload_bytes / self.dense_payload_bytes) * 100

    @property
    def dense_comparison_payload_bytes(self) -> int:
        """Dense wire size used for the equal-payload timing baseline.

        The full Dense parameter remains the savings reference. Timing instead
        compares Sparse against a Dense BF16 buffer with the same wire size,
        so each update rate has a meaningful transport-time baseline.
        """
        return self.sparse_payload_bytes


def build_measurement_rows(
    *,
    dense_payload_bytes: int,
    ratios: Iterable[float],
    value_bytes: int,
    index_bytes: int,
) -> list[SparseTransferMeasurementRow]:
    """Calculate sparse indices-plus-values payloads for one Dense tensor.

    ``dense_payload_bytes`` represents a complete tensor whose elements have
    ``value_bytes`` each. A sparse patch stores one ``index_bytes`` index and
    one value for every selected element. Counts use floor semantics so the
    generated tensor always has a representable whole number of entries.
    """
    if dense_payload_bytes <= 0:
        raise ValueError("dense_payload_bytes must be positive")
    if value_bytes <= 0 or index_bytes <= 0:
        raise ValueError("value_bytes and index_bytes must be positive")
    if dense_payload_bytes % value_bytes:
        raise ValueError("dense_payload_bytes must be divisible by value_bytes")

    total_elements = dense_payload_bytes // value_bytes
    rows = []
    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise ValueError("each ratio must be in (0, 1]")
        update_elements = math.floor(total_elements * ratio)
        if update_elements == 0:
            raise ValueError("ratio selects no elements from the Dense payload")
        rows.append(
            SparseTransferMeasurementRow(
                ratio=ratio,
                dense_payload_bytes=dense_payload_bytes,
                update_elements=update_elements,
                sparse_payload_bytes=update_elements * (index_bytes + value_bytes),
            )
        )
    return rows
