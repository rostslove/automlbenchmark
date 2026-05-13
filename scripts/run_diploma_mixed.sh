#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK="AutoGluon"
CONSTRAINT="test"
MODE="local"
PART="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --framework|-f)
      FRAMEWORK="$2"
      shift 2
      ;;
    --constraint|-c)
      CONSTRAINT="$2"
      shift 2
      ;;
    --mode|-m)
      MODE="$2"
      shift 2
      ;;
    --part|-p)
      PART="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--framework NAME] [--constraint NAME] [--mode local|docker|singularity|aws] [--part all|classification|regression]" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BENCHMARK="diploma_mixed"
TASK_ARGS=()

case "${PART}" in
  all)
    ;;
  classification)
    TASK_ARGS=(-t kc2_binary_classification iris_multiclass_classification)
    ;;
  regression)
    TASK_ARGS=(-t cholesterol_regression autoMpg_regression)
    ;;
  *)
    echo "Invalid part: ${PART}" >&2
    echo "Allowed values: all, classification, regression" >&2
    exit 2
    ;;
esac

python runbenchmark.py "${FRAMEWORK}" "${BENCHMARK}" "${CONSTRAINT}" -m "${MODE}" "${TASK_ARGS[@]}"
