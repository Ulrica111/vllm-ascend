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
    / "benchmark_sparse_hccl_http.py"
)


def _load_benchmark_module():
    benchmark_dir = str(BENCHMARK_PATH.parent)
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    return runpy.run_path(str(BENCHMARK_PATH), run_name="sparse_hccl_benchmark_test")


def test_plan_sparse_update_counts_uses_total_model_elements():
    module = _load_benchmark_module()

    plan = module["plan_sparse_update_counts"](
        [("a", 1_000), ("b", 9_000)], ratio=0.1
    )
    assert plan == [("a", 100), ("b", 900)]


def test_plan_sparse_update_counts_skips_zero_sized_parameter_updates():
    module = _load_benchmark_module()

    plan = module["plan_sparse_update_counts"](
        [("small", 10), ("large", 10_000)], ratio=0.001
    )

    assert plan == [("large", 10)]


def test_completion_payload_uses_greedy_single_token_generation():
    module = _load_benchmark_module()

    payload = module["build_completion_payload"](
        model="Qwen3-4B", prompt="The future of AI is"
    )

    assert payload == {
        "model": "Qwen3-4B",
        "prompt": "The future of AI is",
        "max_tokens": 1,
        "temperature": 0,
    }


def test_benchmark_uses_qwen3_runtime_adapter():
    module = _load_benchmark_module()

    assert module["build_qwen3_runtime_parameters"].__name__ == (
        "build_qwen3_runtime_parameters"
    )


def test_memory_summary_uses_byte_units():
    module = _load_benchmark_module()

    assert module["summarize_bytes"]([10, 20]) == {
        "mean_bytes": 15,
        "median_bytes": 15.0,
        "p90_bytes": 20,
        "min_bytes": 10,
        "max_bytes": 20,
    }


class _Qwen3Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "embedding", torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
        )
        self.register_parameter(
            "q", torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
        )
        self.register_parameter("k", torch.nn.Parameter(torch.tensor([[3.0]])))
        self.register_parameter("v", torch.nn.Parameter(torch.tensor([[4.0]])))
        self.register_parameter("gate", torch.nn.Parameter(torch.tensor([[5.0]])))
        self.register_parameter("up", torch.nn.Parameter(torch.tensor([[6.0]])))
        self.register_parameter("down", torch.nn.Parameter(torch.tensor([[7.0, 8.0]])))

    def named_parameters(self, *args, **kwargs):
        return iter(
            [
                ("model.embed_tokens.weight", self.embedding),
                ("model.layers.0.self_attn.q_proj.weight", self.q),
                ("model.layers.0.self_attn.k_proj.weight", self.k),
                ("model.layers.0.self_attn.v_proj.weight", self.v),
                ("model.layers.0.mlp.gate_proj.weight", self.gate),
                ("model.layers.0.mlp.up_proj.weight", self.up),
                ("model.layers.0.mlp.down_proj.weight", self.down),
            ]
        )


def test_qwen3_runtime_adapter_fuses_qkv_and_gate_up(monkeypatch):
    benchmark_dir = str(BENCHMARK_PATH.parent)
    monkeypatch.syspath_prepend(benchmark_dir)
    sys.modules.pop("qwen3_runtime_weights", None)

    from qwen3_runtime_weights import build_qwen3_runtime_parameters

    runtime = build_qwen3_runtime_parameters(_Qwen3Fixture())

    assert torch.equal(
        runtime["model.layers.0.self_attn.qkv_proj.weight"],
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    assert torch.equal(
        runtime["model.layers.0.mlp.gate_up_proj.weight"],
        torch.tensor([[5.0], [6.0]]),
    )
    assert torch.equal(
        runtime["model.embed_tokens.weight"],
        torch.arange(6, dtype=torch.float32),
    )
    assert torch.equal(
        runtime["model.layers.0.mlp.down_proj.weight"],
        torch.tensor([[7.0, 8.0]]),
    )
