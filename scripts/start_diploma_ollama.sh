#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLM_MODEL:-${OLLAMA_MODEL:-qwen2.5-coder:32b}}"
MODEL_ALIAS="${LLM_MODEL_ALIAS:-${OLLAMA_MODEL_ALIAS:-gpt-4o-mini}}"
MODEL_ALIASES="${OLLAMA_MODEL_ALIASES:-gpt-4o-mini gpt-4o gpt-4 gpt-3.5-turbo gpt-3.5-turbo-16k}"
PORT="${OLLAMA_PORT:-11434}"
GPU="${OLLAMA_USE_GPU:-${OLLAMA_GPU:-0}}"
ENV_FILE="${OLLAMA_ENV_FILE:-scripts/diploma_ollama.env}"
OLLAMA_HOST_URL="${OLLAMA_HOST:-http://127.0.0.1:${PORT}}"
OPENAI_BASE_URL="${OLLAMA_HOST_URL%/}/v1"
EXTERNAL="${OLLAMA_EXTERNAL:-0}"

compose_files=(-f docker-compose.ollama.yml)
if [[ "$GPU" == "1" || "$GPU" == "true" || "$GPU" == "yes" ]]; then
  compose_files+=(-f docker-compose.ollama.gpu.yml)
fi

if [[ "$EXTERNAL" != "1" && "$EXTERNAL" != "true" && "$EXTERNAL" != "yes" ]]; then
  LLM_MODEL="$MODEL" OLLAMA_MODEL="$MODEL" OLLAMA_PORT="$PORT" docker compose "${compose_files[@]}" up -d ollama
  for _ in {1..60}; do
    if LLM_MODEL="$MODEL" OLLAMA_MODEL="$MODEL" OLLAMA_PORT="$PORT" docker compose "${compose_files[@]}" exec -T ollama ollama list >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  if ! LLM_MODEL="$MODEL" OLLAMA_MODEL="$MODEL" OLLAMA_PORT="$PORT" docker compose "${compose_files[@]}" exec -T ollama ollama list >/dev/null 2>&1; then
    echo "Ollama container did not become ready." >&2
    exit 1
  fi

  LLM_MODEL="$MODEL" OLLAMA_MODEL="$MODEL" OLLAMA_PORT="$PORT" docker compose "${compose_files[@]}" run --rm ollama-pull
fi

for _ in {1..60}; do
  if curl -fsS "${OLLAMA_HOST_URL%/}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS "${OLLAMA_HOST_URL%/}/api/tags" >/dev/null 2>&1; then
  echo "Ollama endpoint is not reachable: ${OLLAMA_HOST_URL}" >&2
  exit 1
fi

for alias in $MODEL_ALIAS $MODEL_ALIASES; do
  if command -v ollama >/dev/null 2>&1; then
    if ! OLLAMA_HOST="$OLLAMA_HOST_URL" ollama show "$alias" >/dev/null 2>&1; then
      OLLAMA_HOST="$OLLAMA_HOST_URL" ollama cp "$MODEL" "$alias"
    fi
  elif ! curl -fsS "${OLLAMA_HOST_URL%/}/api/show" -d "{\"model\":\"$alias\"}" >/dev/null 2>&1; then
    curl -fsS "${OLLAMA_HOST_URL%/}/api/create" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$alias\",\"from\":\"$MODEL\",\"stream\":false}" >/dev/null
  fi
done

mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<EOF
unset OPENROUTER_API_KEY
unset OPENROUTER_MODEL
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset https_proxy
unset all_proxy
export AGENT_LLM_BASE_URL="${OPENAI_BASE_URL}"
export AGENT_LLM_API_KEY="ollama"
export AGENT_LLM_MODEL="${MODEL_ALIAS}"
export LLM_MODEL_ALIAS="${MODEL_ALIAS}"
export OLLAMA_MODEL_ALIAS="${MODEL_ALIAS}"
export LLM_MODEL="${MODEL}"
export OLLAMA_MODEL="${MODEL}"
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="${OPENAI_BASE_URL}"
export OPENAI_API_BASE="${OPENAI_BASE_URL}"
export NO_PROXY="127.0.0.1,localhost,ollama,\${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,ollama,\${no_proxy:-}"
EOF

echo
echo "Wrote $ENV_FILE"
echo "Ollama OpenAI-compatible endpoint: ${OPENAI_BASE_URL}"
echo "Model: $MODEL"
echo "Framework model alias: $MODEL_ALIAS"
echo
echo "Run:"
echo "  source $ENV_FILE"
echo "  python scripts/run_diploma_agent_frameworks.py --framework all --fold 0 --setup skip --ollama --continue-on-error"
