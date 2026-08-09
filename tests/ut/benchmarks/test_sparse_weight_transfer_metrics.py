# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fixed-payload sparse transfer benchmark metrics."""

from benchmarks.weight_transfer.sparse_weight_transfer_metrics import (
    build_measurement_rows,
)


def test_fixed_eight_mib_payload_has_expected_sparse_wire_sizes() -> None:
    """A wrong index/value accounting must not silently skew the result table."""
    rows = build_measurement_rows(
        dense_payload_bytes=8 * 2**20,
        ratios=(0.001, 0.01, 0.1, 0.5, 1.0),
        value_bytes=2,
        index_bytes=4,
    )

    assert [row.sparse_payload_bytes for row in rows] == [
        25_164,
        251_658,
        2_516_580,
        12_582_912,
        25_165_824,
    ]
    assert [round(row.savings_percent, 2) for row in rows] == [
        99.7,
        97.0,
        70.0,
        -50.0,
        -200.0,
    ]
    assert [row.dense_comparison_payload_bytes for row in rows] == [
        25_164,
        251_658,
        2_516_580,
        12_582_912,
        25_165_824,
    ]


def test_measurement_rows_reject_payload_incompatible_with_value_dtype() -> None:
    """A non-element-aligned Dense payload cannot define a valid update count."""
    try:
        build_measurement_rows(
            dense_payload_bytes=9,
            ratios=(0.1,),
            value_bytes=2,
            index_bytes=4,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for an unaligned payload")
