#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLM_MODEL:-${OLLAMA_MODEL:-qwen2.5-coder:32b}}"
PORT="${OLLAMA_PORT:-11434}"
GPU="${OLLAMA_GPU:-0}"
ENV_FILE="${OLLAMA_ENV_FILE:-scripts/diploma_ollama.env}"

compose_files=(-f docker-compose.ollama.yml)
if [[ "$GPU" == "1" || "$GPU" == "true" || "$GPU" == "yes" ]]; then
  compose_files+=(-f docker-compose.ollama.gpu.yml)
fi

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
export AGENT_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export AGENT_LLM_API_KEY="ollama"
export AGENT_LLM_MODEL="${MODEL}"
export LLM_MODEL="${MODEL}"
export OLLAMA_MODEL="${MODEL}"
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export NO_PROXY="127.0.0.1,localhost,ollama,\${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,ollama,\${no_proxy:-}"
EOF

echo
echo "Wrote $ENV_FILE"
echo "Ollama OpenAI-compatible endpoint: http://127.0.0.1:${PORT}/v1"
echo "Model: $MODEL"
echo
echo "Run:"
echo "  source $ENV_FILE"
echo "  python scripts/run_diploma_agent_frameworks.py --framework all --fold 0 --setup skip --ollama --continue-on-error"
