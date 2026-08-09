#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the five real-Qwen3 Sparse HCCL update ratios against an already-running
# server configured with {"backend":"sparse_hccl"}. The script intentionally
# launches the Python benchmark once per ratio so every ratio has independent
# trainer loading, five-repeat samples, JSON output, log, and exit status.

set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}
benchmark_py=${BENCHMARK_PY:-"${script_dir}/benchmark_sparse_hccl_http.py"}
model=""
base_url="http://127.0.0.1:8000"
output_dir=""
warmup=1
repeats=5

usage() {
    cat <<'EOF'
Usage: run_sparse_hccl_qwen3.sh --model MODEL --output-dir DIRECTORY [options]

Required:
  --model MODEL              Local Qwen3 checkpoint path or HF model ID.
  --output-dir DIRECTORY     Directory for one JSON/log/status trio per ratio.

Optional:
  --base-url URL             Sparse HCCL server URL (default: http://127.0.0.1:8000).
  --warmup COUNT             Warmup updates per ratio (default: 1).
  --repeats COUNT            Timed updates per ratio (default: 5).
EOF
}

while (($#)); do
    case "$1" in
        --model)
            model=$2
            shift 2
            ;;
        --base-url)
            base_url=$2
            shift 2
            ;;
        --output-dir)
            output_dir=$2
            shift 2
            ;;
        --warmup)
            warmup=$2
            shift 2
            ;;
        --repeats)
            repeats=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$model" || -z "$output_dir" ]]; then
    usage >&2
    exit 2
fi

mkdir -p "$output_dir"

run_ratio() {
    local ratio=$1
    local label=$2
    local prefix="${output_dir}/${label}_sparse_hccl_qwen3_4b"
    local status

    echo "=== Sparse HCCL ratio=${ratio}, warmup=${warmup}, repeats=${repeats} ==="
    "$python_bin" "$benchmark_py" \
        --model "$model" \
        --base-url "$base_url" \
        --ratio "$ratio" \
        --warmup "$warmup" \
        --repeats "$repeats" \
        --output "${prefix}.json" \
        2>&1 | tee "${prefix}.log"
    status=${PIPESTATUS[0]}
    printf '%s\n' "$status" > "${prefix}.status"
    if ((status != 0)); then
        echo "Sparse HCCL ratio=${ratio} failed; stopping." >&2
        exit "$status"
    fi
}

run_ratio 0.001 S014_0p1
run_ratio 0.01 S015_1p0
run_ratio 0.1 S016_10p0
run_ratio 0.5 S017_50p0
run_ratio 1.0 S018_100p0
