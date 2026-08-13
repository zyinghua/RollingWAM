#!/usr/bin/env bash
# Links the repo (editable install) on first container start, then hands over to
# the requested command. Dependencies are already baked into the image, so this
# is offline and takes seconds; restarts skip it once the package imports.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python -c "import rollingwam" >/dev/null 2>&1; then
  echo "[entrypoint] pip install -e ${REPO_DIR} (--no-deps, deps are baked in the image)"
  pip install -e "${REPO_DIR}" --no-deps --no-build-isolation
fi

exec "$@"
