# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert Qwen3 HF parameters to vLLM's TP=1 runtime parameter layout."""

from collections.abc import Iterable

import torch


class Qwen3RuntimeMappingError(ValueError):
    """The HF checkpoint cannot be represented by the Qwen3 runtime layout."""


_QKV_SUFFIXES = ("q_proj.weight", "k_proj.weight", "v_proj.weight")
_GATE_UP_SUFFIXES = ("gate_proj.weight", "up_proj.weight")


def _fuse(
    parameters: dict[str, torch.Tensor],
    names: Iterable[str],
    runtime_name: str,
) -> torch.Tensor:
    tensors = []
    for name in names:
        try:
            tensors.append(parameters[name])
        except KeyError as exc:
            raise Qwen3RuntimeMappingError(
                f"Missing Qwen3 source parameter {name!r} for {runtime_name!r}"
            ) from exc

    first = tensors[0]
    if any(
        tensor.dtype != first.dtype
        or tensor.ndim != first.ndim
        or tensor.shape[1:] != first.shape[1:]
        for tensor in tensors[1:]
    ):
        raise Qwen3RuntimeMappingError(
            f"Cannot fuse incompatible Qwen3 parameters into {runtime_name!r}"
        )
    return torch.cat(tensors, dim=0).contiguous()


def build_qwen3_runtime_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Return every Qwen3 TP=1 runtime parameter from a HF Qwen3 model.

    vLLM combines Q/K/V and gate/up projection weights. Remaining parameters
    retain their HF names and layout at TP=1.
    """
    source = dict(model.named_parameters())
    runtime: dict[str, torch.Tensor] = {}

    for name, parameter in source.items():
        if name.endswith(_QKV_SUFFIXES) or name.endswith(_GATE_UP_SUFFIXES):
            continue
        runtime[name] = parameter.data

    layer_prefixes = {
        name.removesuffix("q_proj.weight")
        for name in source
        if name.endswith("q_proj.weight")
    }
    for prefix in layer_prefixes:
        runtime[f"{prefix}qkv_proj.weight"] = _fuse(
            source,
            [f"{prefix}{suffix}" for suffix in _QKV_SUFFIXES],
            f"{prefix}qkv_proj.weight",
        )

    mlp_prefixes = {
        name.removesuffix("gate_proj.weight")
        for name in source
        if name.endswith("gate_proj.weight")
    }
    for prefix in mlp_prefixes:
        runtime[f"{prefix}gate_up_proj.weight"] = _fuse(
            source,
            [f"{prefix}{suffix}" for suffix in _GATE_UP_SUFFIXES],
            f"{prefix}gate_up_proj.weight",
        )

    if not runtime:
        raise Qwen3RuntimeMappingError("Qwen3 model has no parameters")
    return runtime
