#!/usr/bin/env bash

set -euo pipefail

if [[ "${CODESPACES:-}" != "true" ]]; then
  echo "Not running in GitHub Codespaces; leaving .env unchanged."
  exit 0
fi

codespace_name="${CODESPACE_NAME:?CODESPACE_NAME is required}"
forwarding_domain="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:?GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN is required}"

if [[ ! "${codespace_name}" =~ ^[a-z0-9-]+$ ]]; then
  echo "CODESPACE_NAME contains unexpected characters." >&2
  exit 1
fi

if [[ ! "${forwarding_domain}" =~ ^[a-z0-9.-]+$ ]]; then
  echo "The Codespaces forwarding domain contains unexpected characters." >&2
  exit 1
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${repository_root}/.env"
frontend_url="https://${codespace_name}-5173.${forwarding_domain}"
backend_url="https://${codespace_name}-8000.${forwarding_domain}"

touch "${env_file}"

upsert_env_value() {
  local key="$1"
  local value="$2"

  if grep -q "^${key}=" "${env_file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

ensure_env_value() {
  local key="$1"
  local value="$2"

  if ! grep -q "^${key}=" "${env_file}"; then
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

sed -i '/^VITE_API_URL=/d' "${env_file}"
ensure_env_value "DATABASE_URL" "sqlite:///./app.db"
ensure_env_value "MIN_TRAINING_SAMPLES" "20"
ensure_env_value "ML_TEST_SIZE" "0.2"
ensure_env_value "ML_RANDOM_STATE" "42"
ensure_env_value "MODEL_DIR" "models"
upsert_env_value "FRONTEND_ORIGIN" "${frontend_url}"
upsert_env_value "VITE_API_BASE_URL" "${backend_url}"

echo "Updated .env with this Codespace's frontend and backend URLs."

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is unavailable; port 8000 visibility was not changed." >&2
  exit 0
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  if GH_TOKEN="${GITHUB_TOKEN}" gh codespace ports visibility 8000:public -c "${codespace_name}"; then
    echo "Backend port 8000 is public."
  else
    echo "Could not make port 8000 public. Check the Codespaces organization policy." >&2
  fi
elif gh codespace ports visibility 8000:public -c "${codespace_name}"; then
  echo "Backend port 8000 is public."
else
  echo "Could not make port 8000 public. Check GitHub CLI authentication and the Codespaces organization policy." >&2
fi
