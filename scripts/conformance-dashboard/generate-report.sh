#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_PATH="$ROOT_DIR/IntegrationTesting/IntegrationTesting.xcodeproj"
MANIFEST_PATH="$ROOT_DIR/scripts/conformance-dashboard/conformance-manifest.json"
REPORT_TOOL="$ROOT_DIR/scripts/conformance-dashboard/report.py"

PLATFORM="macOS"
DESTINATION="platform=macOS"
OUTPUT_DIR="$ROOT_DIR/.conformance"
RESULT_BUNDLE=""
DEVELOPMENT_TEAM=""
RUN_NETWORK=0

usage() {
  cat <<'EOF'
Usage: generate-report.sh [options]

Options:
  --platform macOS|iOS-device    Logical report platform (default: macOS).
  --destination DESTINATION     xcodebuild destination. iOS requires a physical device id.
  --development-team TEAM_ID    Signing team for a physical iOS device.
  --result-bundle PATH          Parse an existing .xcresult instead of running tests.
  --output-dir PATH             Output directory (default: .conformance).
  --include-network             Enable pinned checkpoint integration tests.
  --help                        Show this help.

MLX execution on iOS Simulator is intentionally rejected. Use a physical iOS
device destination such as 'platform=iOS,id=<DEVICE_UDID>'.
EOF
}

while (($#)); do
  case "$1" in
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --destination)
      DESTINATION="$2"
      shift 2
      ;;
    --development-team)
      DEVELOPMENT_TEAM="$2"
      shift 2
      ;;
    --result-bundle)
      RESULT_BUNDLE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --include-network)
      RUN_NETWORK=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PLATFORM" in
  macOS)
    if [[ "$DESTINATION" != *"platform=macOS"* ]]; then
      echo "error: macOS reports require a macOS destination" >&2
      exit 2
    fi
    ;;
  iOS-device)
    destination_lower="$(printf '%s' "$DESTINATION" | tr '[:upper:]' '[:lower:]')"
    if [[ "$destination_lower" == *"simulator"* ]]; then
      echo "error: MLX conformance does not support iOS Simulator; use a physical device" >&2
      exit 2
    fi
    if [[ "$DESTINATION" != *"platform=iOS"* || "$DESTINATION" != *"id="* ]]; then
      echo "error: iOS-device requires --destination 'platform=iOS,id=<DEVICE_UDID>'" >&2
      exit 2
    fi
    ;;
  *)
    echo "error: --platform must be macOS or iOS-device" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

XCODE_STATUS=0
if [[ -z "$RESULT_BUNDLE" ]]; then
  RESULT_BUNDLE="$WORK_DIR/conformance.xcresult"
  XCODE_ARGS=(
    test
    -project "$PROJECT_PATH"
    -scheme IntegrationTesting
    -resultBundlePath "$RESULT_BUNDLE"
    -destination "$DESTINATION"
    -skipPackagePluginValidation
    -only-testing:IntegrationTestingTests/VLMConformanceIntegrationTests
    -only-testing:IntegrationTestingTests/LongContextConformanceIntegrationTests
  )
  if [[ -n "$DEVELOPMENT_TEAM" ]]; then
    XCODE_ARGS+=("DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM" CODE_SIGN_STYLE=Automatic)
  fi

  if ((RUN_NETWORK)); then
    export MLX_RUN_CONFORMANCE_NETWORK=1
  fi

  set +e
  xcodebuild "${XCODE_ARGS[@]}"
  XCODE_STATUS=$?
  set -e
fi

if [[ ! -d "$RESULT_BUNDLE" ]]; then
  echo "error: no result bundle produced at $RESULT_BUNDLE" >&2
  exit 1
fi

TESTS_JSON="$WORK_DIR/tests.json"
xcrun xcresulttool get test-results tests \
  --path "$RESULT_BUNDLE" \
  --compact \
  > "$TESTS_JSON"

CHECKED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
python3 "$REPORT_TOOL" \
  --manifest "$MANIFEST_PATH" \
  --tests-json "$TESTS_JSON" \
  --platform "$PLATFORM" \
  --destination "$DESTINATION" \
  --checked-at "$CHECKED_AT" \
  --json-output "$OUTPUT_DIR/conformance-report.json" \
  --markdown-output "$OUTPUT_DIR/conformance-report.md"
REPORT_STATUS=$?
set -e

echo "JSON report: $OUTPUT_DIR/conformance-report.json" >&2
echo "Markdown report: $OUTPUT_DIR/conformance-report.md" >&2

if ((XCODE_STATUS != 0)); then
  echo "error: xcodebuild exited with status $XCODE_STATUS; failures remain in the report" >&2
  exit "$XCODE_STATUS"
fi
exit "$REPORT_STATUS"
