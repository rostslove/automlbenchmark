#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${AGENT_FRAMEWORKS_DIR:-$HOME/agent-frameworks}"
VENV_DIR="${AGENT_FRAMEWORKS_VENV_DIR:-$HOME/agent-framework-venvs}"
PYTHON_BIN="${PYTHON:-python}"
ENV_FILE="${ENV_FILE:-scripts/diploma_agent_frameworks.env}"

clone_or_update() {
  local repo_url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "Updating $dest"
    git -C "$dest" pull --ff-only || echo "Could not fast-forward $dest; keeping existing checkout."
  elif [[ -e "$dest" ]]; then
    echo "Using existing $dest"
  else
    echo "Cloning $repo_url -> $dest"
    git clone "$repo_url" "$dest"
  fi
}

create_venv() {
  local name="$1"
  local venv="$VENV_DIR/$name"
  if [[ ! -x "$venv/bin/python" ]]; then
    echo "Creating venv $venv" >&2
    "$PYTHON_BIN" -m venv "$venv"
  fi
  "$venv/bin/python" -m pip install -U pip uv >&2
  echo "$venv"
}

install_requirements_filtered() {
  local python="$1"
  local requirements="$2"
  local tmp
  tmp="$(mktemp)"
  # AutoML-Agent currently pins python_dateutil twice to mutually exclusive values.
  grep -v -E '^(python[-_]dateutil==2\.9\.0)$' "$requirements" > "$tmp"
  "$python" -m pip install -r "$tmp" || {
    echo "WARNING: requirements install failed for $requirements; continuing with partial environment." >&2
  }
  rm -f "$tmp"
}

mkdir -p "$ROOT_DIR" "$VENV_DIR"

echo "Installing CLI agent frameworks in an isolated venv..."
CLI_VENV="$(create_venv cli)"
"$CLI_VENV/bin/python" -m uv pip install -U "autogluon.assistant>=1.0" aideml

AUTOKAGGLE_REPO="$ROOT_DIR/AutoKaggle"
AUTOML_AGENT_REPO="$ROOT_DIR/automl-agent"
DS_AGENT_REPO="$ROOT_DIR/DS-Agent"

clone_or_update "https://github.com/multimodal-art-projection/AutoKaggle.git" "$AUTOKAGGLE_REPO"
clone_or_update "https://github.com/DeepAuto-AI/automl-agent.git" "$AUTOML_AGENT_REPO"
clone_or_update "https://github.com/guosyjlu/DS-Agent.git" "$DS_AGENT_REPO"

echo "Installing repository framework requirements in isolated venvs..."
AUTOKAGGLE_VENV="$(create_venv autokaggle)"
if [[ -f "$AUTOKAGGLE_REPO/requirements.txt" ]]; then
  install_requirements_filtered "$AUTOKAGGLE_VENV/bin/python" "$AUTOKAGGLE_REPO/requirements.txt"
fi

AUTOML_AGENT_VENV="$(create_venv automl-agent)"
if [[ -f "$AUTOML_AGENT_REPO/requirements.txt" ]]; then
  install_requirements_filtered "$AUTOML_AGENT_VENV/bin/python" "$AUTOML_AGENT_REPO/requirements.txt"
fi

DS_AGENT_VENV="$(create_venv ds-agent)"
if [[ -d "$DS_AGENT_REPO/development" ]]; then
  "$DS_AGENT_VENV/bin/python" -m pip install -e "$DS_AGENT_REPO/development" || true
  if [[ -f "$DS_AGENT_REPO/development/requirements.txt" ]]; then
    install_requirements_filtered "$DS_AGENT_VENV/bin/python" "$DS_AGENT_REPO/development/requirements.txt"
  fi
fi

mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<EOF
export MLZERO_COMMAND="$CLI_VENV/bin/mlzero"
export AIDE_PYTHON="$CLI_VENV/bin/python"
export AUTOKAGGLE_REPO="$AUTOKAGGLE_REPO"
export AUTOML_AGENT_REPO="$AUTOML_AGENT_REPO"
export DS_AGENT_REPO="$DS_AGENT_REPO"
export AUTOKAGGLE_PYTHON="$AUTOKAGGLE_VENV/bin/python"
export AUTOML_AGENT_PYTHON="$AUTOML_AGENT_VENV/bin/python"
export DS_AGENT_PYTHON="$DS_AGENT_VENV/bin/python"
EOF

if [[ -x "$CLI_VENV/bin/aide" ]]; then
  echo "export AIDE_COMMAND=\"$CLI_VENV/bin/aide\"" >> "$ENV_FILE"
fi

echo
echo "Wrote $ENV_FILE"
echo "Run:"
echo "  source $ENV_FILE"
echo "  python scripts/run_diploma_agent_frameworks.py --framework all --fold 0 --setup skip --continue-on-error"
