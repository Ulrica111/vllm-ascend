#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run real-Qwen3 Sparse NPU IPC measurements for five update rates. The vLLM
# server must already be running with {"backend":"sparse_ipc"} on the same
# physical NPU as this trainer process.

set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}
benchmark_py=${BENCHMARK_PY:-"${script_dir}/benchmark_sparse_npu_ipc_http.py"}
model=""
base_url="http://127.0.0.1:8000"
output_dir=""
device=0
warmup=1
repeats=5
max_updates_per_request=16000000

usage() {
    echo "Usage: $0 --model MODEL --output-dir DIRECTORY" >&2
    echo "       [--base-url URL] [--device N] [--warmup N] [--repeats N]" >&2
    echo "       [--max-updates-per-request N]" >&2
}

while (($#)); do
    case "$1" in
        --model) model=$2; shift 2 ;;
        --base-url) base_url=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --device) device=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --max-updates-per-request) max_updates_per_request=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$model" || -z "$output_dir" ]]; then
    usage
    exit 2
fi

mkdir -p "$output_dir"

run_ratio() {
    local ratio=$1
    local label=$2
    local prefix="${output_dir}/${label}_sparse_ipc_qwen3_4b"
    local status

    echo "=== Sparse IPC ratio=${ratio}, warmup=${warmup}, repeats=${repeats} ==="
    "$python_bin" "$benchmark_py" \
        --model "$model" \
        --base-url "$base_url" \
        --device "$device" \
        --ratio "$ratio" \
        --warmup "$warmup" \
        --repeats "$repeats" \
        --max-updates-per-request "$max_updates_per_request" \
        --output "${prefix}.json" \
        2>&1 | tee "${prefix}.log"
    status=${PIPESTATUS[0]}
    printf '%s\n' "$status" > "${prefix}.status"
    if ((status != 0)); then
        echo "Sparse IPC ratio=${ratio} failed; stopping." >&2
        exit "$status"
    fi
}

run_ratio 0.001 S019_0p1
run_ratio 0.01 S020_1p0
run_ratio 0.1 S021_10p0
run_ratio 0.5 S022_50p0
run_ratio 1.0 S023_100p0
