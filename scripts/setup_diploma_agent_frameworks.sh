#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${AGENT_FRAMEWORKS_DIR:-$HOME/agent-frameworks}"
PYTHON_BIN="${PYTHON:-python}"
ENV_FILE="${ENV_FILE:-scripts/diploma_agent_frameworks.env}"

clone_or_update() {
  local repo_url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "Updating $dest"
    git -C "$dest" pull --ff-only
  elif [[ -e "$dest" ]]; then
    echo "Using existing $dest"
  else
    echo "Cloning $repo_url -> $dest"
    git clone "$repo_url" "$dest"
  fi
}

echo "Installing CLI agent frameworks into the active Python environment..."
"$PYTHON_BIN" -m pip install -U pip uv
"$PYTHON_BIN" -m uv pip install -U "autogluon.assistant>=1.0" aideml

mkdir -p "$ROOT_DIR"

AUTOKAGGLE_REPO="$ROOT_DIR/AutoKaggle"
AUTOML_AGENT_REPO="$ROOT_DIR/automl-agent"
DS_AGENT_REPO="$ROOT_DIR/DS-Agent"

clone_or_update "https://github.com/multimodal-art-projection/AutoKaggle.git" "$AUTOKAGGLE_REPO"
clone_or_update "https://github.com/DeepAuto-AI/automl-agent.git" "$AUTOML_AGENT_REPO"
clone_or_update "https://github.com/guosyjlu/DS-Agent.git" "$DS_AGENT_REPO"

echo "Installing repository framework requirements..."
if [[ -f "$AUTOKAGGLE_REPO/requirements.txt" ]]; then
  "$PYTHON_BIN" -m pip install -r "$AUTOKAGGLE_REPO/requirements.txt"
fi
if [[ -f "$AUTOML_AGENT_REPO/requirements.txt" ]]; then
  "$PYTHON_BIN" -m pip install -r "$AUTOML_AGENT_REPO/requirements.txt"
fi
if [[ -d "$DS_AGENT_REPO/development" ]]; then
  "$PYTHON_BIN" -m pip install -e "$DS_AGENT_REPO/development"
  if [[ -f "$DS_AGENT_REPO/development/requirements.txt" ]]; then
    "$PYTHON_BIN" -m pip install -r "$DS_AGENT_REPO/development/requirements.txt"
  fi
fi

mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<EOF
export AUTOKAGGLE_REPO="$AUTOKAGGLE_REPO"
export AUTOML_AGENT_REPO="$AUTOML_AGENT_REPO"
export DS_AGENT_REPO="$DS_AGENT_REPO"
EOF

echo
echo "Wrote $ENV_FILE"
echo "Run:"
echo "  source $ENV_FILE"
echo "  python scripts/run_diploma_agent_frameworks.py --framework all --setup skip --continue-on-error"
