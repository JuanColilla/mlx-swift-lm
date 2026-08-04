import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("conformance_report", ROOT / "report.py")
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


class ConformanceReportTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "conformance-manifest.json").read_text())
        payload = json.loads((ROOT / "fixtures/xcresult-tests.json").read_text())
        self.results = REPORT.parse_xcresult_tests(payload)

    def test_parser_preserves_all_statuses_and_unmapped_failures(self):
        report = REPORT.build_report(
            self.manifest,
            self.results,
            platform="macOS",
            destination="platform=macOS",
            checked_at="2026-07-15T00:00:00Z",
        )

        by_id = {row["check_id"]: row for row in report["checks"]}
        self.assertEqual(by_id["smolvlm2-vlm-image"]["status"], "pass")
        self.assertEqual(by_id["qwen35-long-context"]["status"], "fail")
        self.assertEqual(by_id["fastvlm-device-checkpoint"]["status"], "disabled")
        self.assertEqual(by_id["idefics3-multi-image-device-checkpoint"]["status"], "disabled")
        self.assertEqual(report["unmapped_failures"][0]["source_test"], "UnmappedSuite/unmappedFailure")

    def test_runtime_skips_remain_not_run_instead_of_becoming_manifest_disabled(self):
        payload = {
            "testNodes": [
                {
                    "nodeType": "Test Case",
                    "name": "smolVLM2ImageGeneration()",
                    "nodeIdentifier": "VLMConformanceIntegrationTests/smolVLM2ImageGeneration()",
                    "result": "Skipped",
                }
            ]
        }

        results = REPORT.parse_xcresult_tests(payload)
        self.assertEqual(results[0].status, "not_run")

    def test_missing_enabled_test_is_not_run(self):
        report = REPORT.build_report(
            self.manifest,
            [],
            platform="iOS-device",
            destination="platform=iOS,id=DEVICE",
            checked_at="2026-07-15T00:00:00Z",
        )

        statuses = {row["check_id"]: row["status"] for row in report["checks"]}
        self.assertEqual(statuses["smolvlm2-loads"], "not_run")
        self.assertEqual(statuses["qwen35-long-context"], "not_run")
        self.assertEqual(statuses["gemma4-video-device-checkpoint"], "disabled")

    def test_json_and_markdown_are_stable(self):
        report = REPORT.build_report(
            self.manifest,
            self.results,
            platform="macOS",
            destination="platform=macOS",
            checked_at="2026-07-15T00:00:00Z",
        )

        self.assertEqual(REPORT.render_json(report), REPORT.render_json(report))
        markdown = REPORT.render_markdown(report)
        self.assertEqual(markdown, REPORT.render_markdown(report))
        self.assertIn("| model_id | revision | architecture | capability |", markdown)
        self.assertIn("## Unmapped failures", markdown)

    def test_manifest_rejects_duplicate_check_ids(self):
        duplicate = dict(self.manifest)
        duplicate["checks"] = list(self.manifest["checks"]) + [self.manifest["checks"][0]]
        with self.assertRaisesRegex(ValueError, "duplicate manifest check id"):
            REPORT.build_report(
                duplicate,
                self.results,
                platform="macOS",
                destination="platform=macOS",
                checked_at="2026-07-15T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
