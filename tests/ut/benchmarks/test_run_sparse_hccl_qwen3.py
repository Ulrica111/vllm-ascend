# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the real-Qwen Sparse HCCL benchmark loop wrapper."""

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3]
    / "benchmarks"
    / "weight_transfer"
    / "run_sparse_hccl_qwen3.sh"
)


def test_wrapper_runs_all_five_ratios_and_archives_each_status(tmp_path: Path) -> None:
    """Dropping a ratio must fail this test rather than silently skip a result."""
    fake_python = tmp_path / "fake-python.sh"
    calls = tmp_path / "calls.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_CALLS\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_benchmark = tmp_path / "fake-benchmark.py"
    fake_benchmark.touch()

    env = os.environ | {
        "PYTHON_BIN": str(fake_python),
        "BENCHMARK_PY": str(fake_benchmark),
        "FAKE_CALLS": str(calls),
    }
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model",
            "/weights/Qwen3-4B",
            "--base-url",
            "http://127.0.0.1:8000",
            "--output-dir",
            str(tmp_path / "archive"),
            "--warmup",
            "1",
            "--repeats",
            "5",
        ],
        check=True,
        env=env,
        text=True,
    )

    assert [line.split(" --ratio ")[1].split()[0] for line in calls.read_text().splitlines()] == [
        "0.001",
        "0.01",
        "0.1",
        "0.5",
        "1.0",
    ]
    statuses = sorted((tmp_path / "archive").glob("*.status"))
    assert len(statuses) == 5
    assert all(status.read_text(encoding="utf-8").strip() == "0" for status in statuses)
