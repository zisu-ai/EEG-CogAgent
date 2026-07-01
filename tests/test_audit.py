import tempfile
import unittest
from pathlib import Path

import pandas as pd

from eeg_cogagent.audit import audit_run


class AuditTests(unittest.TestCase):
    def test_complete_minimal_run_passes_required_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bids = root / "data"
            output = root / "results"
            bids.mkdir()
            output.mkdir()
            pd.DataFrame({"participant_id": ["sub-001", "sub-002"]}).to_csv(
                bids / "participants.tsv", sep="\t", index=False
            )
            features = pd.DataFrame({
                "participant_id": ["sub-001", "sub-002"],
                "label": ["AD", "HC"],
                "band_global__alpha": [1.0, 2.0],
                "n_epochs": [10, 11],
            })
            features.to_csv(output / "features.csv", index=False)
            pd.DataFrame({"group": ["AD", "HC"], "n": [1, 1]}).to_csv(
                output / "table1_baseline.csv", index=False
            )
            pd.DataFrame({"feature": ["band_global__alpha"], "q_value": [0.2]}).to_csv(
                output / "feature_statistics.csv", index=False
            )
            pd.DataFrame({
                "model": ["logistic_regression"],
                "accuracy": [0.5],
                "balanced_accuracy": [0.5],
                "auc_ovr": [0.5],
            }).to_csv(output / "model_metrics.csv", index=False)
            pd.DataFrame({
                "model": ["logistic_regression", "logistic_regression"],
                "participant_id": ["sub-001", "sub-002"],
            }).to_csv(output / "model_predictions.csv", index=False)
            (output / "auto_report.md").write_text("report", encoding="utf-8")
            (output / "agent_plan.md").write_text("plan", encoding="utf-8")
            cfg = {
                "project": {"dataset_id": "test", "task": "rest"},
                "participants": {"file": "participants.tsv"},
                "paths": {"bids_root": "data", "output_dir": "results"},
                "_project_root": str(root),
            }

            result = audit_run(cfg)

            self.assertEqual(result["status_counts"]["fail"], 0)
            self.assertEqual(result["summary"]["processed_participants"], 2)
            coverage = next(check for check in result["checks"] if check["name"] == "models:out-of-fold-coverage")
            self.assertEqual(coverage["status"], "pass")


if __name__ == "__main__":
    unittest.main()
