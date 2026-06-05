#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/Magic123"

if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

cd "${REPO}"
source install.sh

echo "Magic123 install script completed. Follow its terminal output for the activated environment."
