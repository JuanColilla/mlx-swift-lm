#!/usr/bin/env bash
#
# Genera un reporte de conformidad (DOCS/conformance-dashboard.md) a partir
# de una ejecución real de la suite de integración.
#
# ADVERTENCIA: ejecutar este script dispara `xcodebuild test` contra el
# scheme `IntegrationTesting`, que descarga checkpoints reales de Hugging
# Face (varios GB en total) la primera vez que corre. No lo ejecutes en CI
# ni en una máquina con ancho de banda/disco limitado sin saberlo.
#
# Uso:
#   scripts/conformance-dashboard/generate-report.sh [output.md]
#
# Requiere: xcodebuild, xcrun xcresulttool, jq.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_PATH="$ROOT_DIR/IntegrationTesting/IntegrationTesting.xcodeproj"
SCHEME="IntegrationTesting"
OUTPUT_PATH="${1:-$ROOT_DIR/conformance-report.md}"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

RESULT_BUNDLE="$WORK_DIR/results.xcresult"

echo "Running $SCHEME test suite (this downloads real model checkpoints)..." >&2

xcodebuild test \
  -project "$PROJECT_PATH" \
  -scheme "$SCHEME" \
  -resultBundlePath "$RESULT_BUNDLE" \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  || true # capture failing tests too; status is read from the result bundle, not the exit code

if [[ ! -d "$RESULT_BUNDLE" ]]; then
  echo "error: no result bundle produced at $RESULT_BUNDLE" >&2
  exit 1
fi

TESTS_JSON="$WORK_DIR/tests.json"
xcrun xcresulttool get test-results tests \
  --path "$RESULT_BUNDLE" \
  --compact \
  > "$TESTS_JSON"

# Flatten the recursive TestNode tree down to individual "Test Case" nodes,
# carrying the enclosing "Test Suite" name (if any) so source_test can be
# emitted as "Suite/testName" to match DOCS/conformance-dashboard.md §2.
jq -r '
  def walk_nodes(suite):
    if .nodeType == "Test Suite" then
      .name as $suite
      | (.children // [])[] | walk_nodes($suite)
    elif .nodeType == "Test Case" then
      {
        source_test: (if suite != null then suite + "/" + .name else .name end),
        status: (
          if .result == "Passed" then "pass"
          elif .result == "Failed" then "fail"
          else "not_run"
          end
        )
      }
    else
      (.children // [])[] | walk_nodes(suite)
    end;
  .testNodes[] | walk_nodes(null)
' "$TESTS_JSON" > "$WORK_DIR/rows.jsonl"

TODAY="$(date -u +%Y-%m-%d)"

{
  echo "# Conformance report"
  echo
  echo "Generated: $TODAY"
  echo "Platform: macOS"
  echo
  echo "| model_id | architecture | capability | platform | status | last_checked | source_test |"
  echo "|---|---|---|---|---|---|---|"
  echo
  echo "<!--"
  echo "This script emits raw per-test pass/fail rows below; mapping each"
  echo "source_test to (model_id, architecture, capability) per the schema in"
  echo "DOCS/conformance-dashboard.md §3 is intentionally left as a manual or"
  echo "follow-up step — the mapping is data (see §3 table), not something to"
  echo "infer from the test name alone, and is out of scope for this scaffold."
  echo "-->"
  echo
  echo "Raw test results (source_test, status, last_checked):"
  echo
  jq -r --arg today "$TODAY" '"| " + .source_test + " | " + .status + " | " + $today + " |"' "$WORK_DIR/rows.jsonl"
} > "$OUTPUT_PATH"

echo "Report written to $OUTPUT_PATH" >&2
