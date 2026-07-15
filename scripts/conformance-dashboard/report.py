#!/usr/bin/env python3
"""Convert xcresult test JSON plus a checked-in manifest into stable reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VALID_STATUSES = {"pass", "fail", "not_run", "disabled"}
RESULT_STATUSES = {
    "passed": "pass",
    "success": "pass",
    "failed": "fail",
    "failure": "fail",
    "skipped": "not_run",
    "disabled": "disabled",
    "not run": "not_run",
    "not_run": "not_run",
}
STATUS_PRIORITY = {"fail": 3, "pass": 2, "disabled": 1, "not_run": 0}


@dataclass(frozen=True)
class TestResult:
    identifier: str
    status: str
    raw_result: str


def canonical_identifier(identifier: str) -> str:
    value = identifier.strip().replace("()", "")
    components = [component for component in value.split("/") if component]
    if len(components) > 2:
        components = components[-2:]
    return "/".join(components)


def _walk_test_nodes(nodes: Iterable[dict[str, Any]], suites: tuple[str, ...] = ()):
    for node in nodes:
        node_type = str(node.get("nodeType", ""))
        name = str(node.get("name", ""))
        next_suites = suites
        if node_type == "Test Suite" and name:
            next_suites = suites + (name,)
        if node_type == "Test Case":
            explicit = node.get("nodeIdentifier") or node.get("identifier")
            if explicit:
                identifier = str(explicit)
            elif suites:
                identifier = f"{suites[-1]}/{name}"
            else:
                identifier = name
            raw_result = str(node.get("result", "Not Run"))
            status = RESULT_STATUSES.get(raw_result.casefold(), "not_run")
            yield TestResult(canonical_identifier(identifier), status, raw_result)
        children = node.get("children") or []
        if isinstance(children, list):
            yield from _walk_test_nodes(children, next_suites)


def parse_xcresult_tests(payload: dict[str, Any]) -> list[TestResult]:
    nodes = payload.get("testNodes") or payload.get("children") or []
    if not isinstance(nodes, list):
        raise ValueError("xcresult JSON must contain a testNodes array")
    return list(_walk_test_nodes(nodes))


def _validated_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ValueError("manifest checks must be an array")
    ids: set[str] = set()
    required = {
        "id",
        "test_identifier",
        "model_id",
        "revision",
        "architecture",
        "capability",
        "platforms",
        "enabled",
    }
    for check in checks:
        missing = required - set(check)
        if missing:
            raise ValueError(f"manifest check is missing fields: {sorted(missing)}")
        if check["id"] in ids:
            raise ValueError(f"duplicate manifest check id: {check['id']}")
        ids.add(check["id"])
        if not isinstance(check["platforms"], list) or not check["platforms"]:
            raise ValueError(f"manifest check {check['id']} needs at least one platform")
        if not check["enabled"] and not check.get("disabled_reason"):
            raise ValueError(f"disabled manifest check {check['id']} needs disabled_reason")
    return payload


def _index_results(results: list[TestResult]) -> dict[str, TestResult]:
    indexed: dict[str, TestResult] = {}
    for result in results:
        existing = indexed.get(result.identifier)
        if existing is None or STATUS_PRIORITY[result.status] > STATUS_PRIORITY[existing.status]:
            indexed[result.identifier] = result
    return indexed


def build_report(
    manifest: dict[str, Any],
    results: list[TestResult],
    *,
    platform: str,
    destination: str,
    checked_at: str,
) -> dict[str, Any]:
    manifest = _validated_manifest(manifest)
    result_index = _index_results(results)
    mapped_identifiers: set[str] = set()
    rows: list[dict[str, Any]] = []

    for check in manifest["checks"]:
        if platform not in check["platforms"]:
            continue
        identifier = canonical_identifier(check["test_identifier"])
        result = result_index.get(identifier)
        if not check["enabled"]:
            status = "disabled"
            details = check["disabled_reason"]
            last_checked = None
        elif result is None:
            status = "not_run"
            details = "No matching test case was present in the xcresult payload."
            last_checked = None
        else:
            mapped_identifiers.add(identifier)
            status = result.status
            details = f"xcresult: {result.raw_result}"
            last_checked = checked_at
        if status not in VALID_STATUSES:
            raise AssertionError(f"invalid status produced: {status}")
        rows.append(
            {
                "architecture": check["architecture"],
                "capability": check["capability"],
                "check_id": check["id"],
                "details": details,
                "last_checked": last_checked,
                "model_id": check["model_id"],
                "platform": platform,
                "revision": check["revision"],
                "source_test": identifier,
                "status": status,
            }
        )

    rows.sort(key=lambda row: (row["model_id"], row["capability"], row["check_id"]))
    unmapped_failures = sorted(
        (
            {
                "source_test": result.identifier,
                "status": result.status,
                "raw_result": result.raw_result,
            }
            for result in results
            if result.status == "fail" and result.identifier not in mapped_identifiers
        ),
        key=lambda row: row["source_test"],
    )
    summary = {status: sum(row["status"] == status for row in rows) for status in sorted(VALID_STATUSES)}

    return {
        "schema_version": 1,
        "generated_at": checked_at,
        "run": {
            "destination": destination,
            "platform": platform,
            "test_cases_observed": len(results),
        },
        "summary": summary,
        "checks": rows,
        "unmapped_failures": unmapped_failures,
    }


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _escape_markdown(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    lines = [
        "# Conformance report",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Platform: `{run['platform']}`  ",
        f"Destination: `{_escape_markdown(run['destination'])}`",
        "",
        "| model_id | revision | architecture | capability | platform | status | last_checked | source_test |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["checks"]:
        values = [
            row["model_id"],
            row["revision"],
            row["architecture"],
            row["capability"],
            row["platform"],
            row["status"],
            row["last_checked"],
            row["source_test"],
        ]
        lines.append("| " + " | ".join(_escape_markdown(value) for value in values) + " |")

    lines.extend(["", "## Summary", ""])
    for status in ("pass", "fail", "not_run", "disabled"):
        lines.append(f"- `{status}`: {report['summary'][status]}")

    if report["unmapped_failures"]:
        lines.extend(
            [
                "",
                "## Unmapped failures",
                "",
                "These failures are intentionally surfaced even though the manifest does not map them:",
                "",
            ]
        )
        for failure in report["unmapped_failures"]:
            lines.append(
                f"- `{_escape_markdown(failure['source_test'])}`: "
                f"`{_escape_markdown(failure['raw_result'])}`"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tests-json", type=Path, required=True)
    parser.add_argument("--platform", choices=("macOS", "iOS-device"), required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--checked-at",
        default=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tests_payload = json.loads(args.tests_json.read_text(encoding="utf-8"))
    report = build_report(
        manifest,
        parse_xcresult_tests(tests_payload),
        platform=args.platform,
        destination=args.destination,
        checked_at=args.checked_at,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(render_json(report), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    has_mapped_failure = any(check["status"] == "fail" for check in report["checks"])
    return 1 if has_mapped_failure or report["unmapped_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
