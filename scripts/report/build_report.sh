#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${ROOT_DIR}/report"
BUILD_DIR="${REPORT_DIR}/build"
mkdir -p "${BUILD_DIR}"
cp "${REPORT_DIR}/references.bib" "${BUILD_DIR}/references.bib"

cd "${REPORT_DIR}"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory "${BUILD_DIR}" main.tex
if [[ -f "${BUILD_DIR}/main.aux" ]]; then
  (cd "${BUILD_DIR}" && bibtex main) || true
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "${BUILD_DIR}" main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "${BUILD_DIR}" main.tex
fi

echo "Report PDF: ${BUILD_DIR}/main.pdf"
