# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
import sys
from pathlib import Path

import torch


BENCHMARK_PATH = (
    Path(__file__).parents[3]
    / "benchmarks"
    / "weight_transfer"
    / "benchmark_dense_hccl_http.py"
)


def _load_benchmark_module():
    benchmark_dir = str(BENCHMARK_PATH.parent)
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    return runpy.run_path(str(BENCHMARK_PATH), run_name="dense_hccl_benchmark_test")


def test_dense_packed_benchmark_uses_256_mib_buffer():
    module = _load_benchmark_module()

    assert module["PACKED_BUFFER_SIZE_BYTES"] == 256 * 2**20


def test_dense_packed_update_info_describes_all_parameters():
    module = _load_benchmark_module()
    parameters = {
        "first.weight": torch.empty((2, 3), dtype=torch.bfloat16),
        "second.weight": torch.empty(7, dtype=torch.float32),
    }
    update_info = module["build_update_info"](parameters)

    assert update_info == {
        "names": ["first.weight", "second.weight"],
        "dtype_names": ["bfloat16", "float32"],
        "shapes": [[2, 3], [7]],
        "packed": True,
        "packed_buffer_size_bytes": 256 * 2**20,
    }
