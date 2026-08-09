# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.distributed.weight_transfer.memory_stats import (
    capture_memory_baseline,
    collect_peak_memory_stats,
)


class FakeMemoryManager:
    def __init__(self):
        self.allocated = 100
        self.reserved = 200
        self.peak_allocated = 100
        self.peak_reserved = 200
        self.did_reset = False

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return self.reserved

    def max_memory_allocated(self):
        return self.peak_allocated

    def max_memory_reserved(self):
        return self.peak_reserved

    def reset_peak_memory_stats(self):
        self.did_reset = True


def test_peak_memory_stats_include_baseline_and_increment():
    memory = FakeMemoryManager()

    baseline = capture_memory_baseline(memory)
    memory.peak_allocated = 150
    memory.peak_reserved = 260

    assert baseline == {"allocated_bytes": 100, "reserved_bytes": 200}
    assert collect_peak_memory_stats(memory, baseline) == {
        "baseline_allocated_bytes": 100,
        "baseline_reserved_bytes": 200,
        "peak_allocated_bytes": 150,
        "peak_reserved_bytes": 260,
        "incremental_allocated_bytes": 50,
        "incremental_reserved_bytes": 60,
    }


def test_capture_memory_baseline_resets_peak_counters():
    memory = FakeMemoryManager()

    capture_memory_baseline(memory)

    assert memory.did_reset
