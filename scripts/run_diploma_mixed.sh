#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK="AutoGluon"
CONSTRAINT="test"
MODE="local"
PART="all"
SETUP="auto"
DEFAULT_FRAMEWORKS=(
  AutoGluon
  flaml
  H2OAutoML
  lightautoml
  mljarsupervised
  TPOT
  RandomForest
)

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
    --setup|-s)
      SETUP="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--framework NAME|all|name1,name2] [--constraint NAME] [--mode local|docker|singularity|aws] [--part all|classification|regression] [--setup auto|skip|force|only]" >&2
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
    TASK_ARGS=(-t kc2_binary_classification iris_multiclass_classification credit_g_binary_classification)
    ;;
  regression)
    TASK_ARGS=(-t cholesterol_regression autoMpg_regression kin8nm_regression)
    ;;
  *)
    echo "Invalid part: ${PART}" >&2
    echo "Allowed values: all, classification, regression" >&2
    exit 2
    ;;
esac

FRAMEWORKS=("${FRAMEWORK}")
if [[ "${FRAMEWORK}" == "all" ]]; then
  FRAMEWORKS=("${DEFAULT_FRAMEWORKS[@]}")
elif [[ "${FRAMEWORK}" == *","* ]]; then
  IFS="," read -r -a FRAMEWORKS <<< "${FRAMEWORK}"
fi

FAILED_FRAMEWORKS=()
for FW in "${FRAMEWORKS[@]}"; do
  FW="$(echo "${FW}" | xargs)"
  echo
  echo "===== Running ${FW} on ${BENCHMARK} (${PART}, ${CONSTRAINT}, ${MODE}, setup=${SETUP}) ====="
  if python runbenchmark.py "${FW}" "${BENCHMARK}" "${CONSTRAINT}" -m "${MODE}" -s "${SETUP}" "${TASK_ARGS[@]}"; then
    echo "===== ${FW}: OK ====="
  else
    echo "===== ${FW}: FAILED =====" >&2
    FAILED_FRAMEWORKS+=("${FW}")
  fi
done

if [[ ${#FAILED_FRAMEWORKS[@]} -gt 0 ]]; then
  echo
  echo "Failed frameworks: ${FAILED_FRAMEWORKS[*]}" >&2
  exit 1
fi
