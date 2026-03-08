"""Evidently AI Data Drift Monitor for Sales Intelligence Hub.

Compares current data distributions against a reference baseline
to detect concept drift and data quality issues.

Compatible with Evidently v0.7+ API.
"""
import os
import sys
import logging
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Evidently v0.7+ imports
from evidently import Report
from evidently.presets import DataDriftPreset


class DriftMonitor:
    """Monitors data distributions for drift using Evidently AI."""

    REPORTS_DIR = os.path.join(settings.BASE_DIR, "reports")

    def __init__(self):
        os.makedirs(self.REPORTS_DIR, exist_ok=True)

    def _load_current_data(self, table: str = "leads") -> pd.DataFrame:
        """Load current data from the database."""
        from sqlalchemy import create_engine
        engine = create_engine(settings.DATABASE_URL)
        query = f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT 5000"
        try:
            return pd.read_sql(query, engine)
        except Exception as e:
            logger.warning(f"Could not load {table}: {e}")
            return pd.read_sql(f"SELECT * FROM {table} LIMIT 5000", engine)

    def save_reference(self, table: str = "leads"):
        """Save current data as the reference baseline."""
        df = self._load_current_data(table)
        ref_path = os.path.join(self.REPORTS_DIR, f"reference_{table}.parquet")
        df.to_parquet(ref_path, index=False)
        logger.info(f"Reference baseline saved: {len(df)} rows from '{table}'")
        return len(df)

    def run_drift_check(self, table: str = "leads") -> dict:
        """Run drift analysis comparing current data against reference."""
        ref_path = os.path.join(self.REPORTS_DIR, f"reference_{table}.parquet")

        # Create reference if it doesn't exist
        if not os.path.exists(ref_path):
            logger.info("No reference baseline found — creating from current data...")
            self.save_reference(table)
            return {
                "status": "baseline_created",
                "message": "Reference baseline created. Run again after new data arrives to check for drift.",
                "table": table,
                "timestamp": datetime.now().isoformat(),
            }

        # Load reference and current data
        reference_data = pd.read_parquet(ref_path)
        current_data = self._load_current_data(table)

        # Select only numeric and categorical columns for comparison
        exclude_cols = [c for c in current_data.columns if c.endswith("_id") or c.endswith("_at") or c.endswith("_date")]
        compare_cols = [c for c in current_data.columns if c not in exclude_cols and c in reference_data.columns]

        if not compare_cols:
            return {
                "status": "error",
                "message": "No comparable columns found between reference and current data.",
                "table": table,
            }

        ref_subset = reference_data[compare_cols].copy()
        cur_subset = current_data[compare_cols].copy()

        # Build and run the Evidently report (v0.7 API)
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=ref_subset, current_data=cur_subset)

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.REPORTS_DIR, f"drift_report_{table}_{timestamp}.html")
        try:
            html_content = snapshot.get_html_str(as_iframe=False)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Drift report saved: {report_path}")
        except Exception as e:
            logger.warning(f"Could not save HTML report: {e}")
            report_path = None

        # Build summary
        drift_summary = {
            "status": "ok",
            "table": table,
            "timestamp": datetime.now().isoformat(),
            "report_path": report_path,
            "reference_rows": len(ref_subset),
            "current_rows": len(cur_subset),
            "columns_analyzed": compare_cols,
            "drift_detected": False,
            "drifted_columns": [],
        }

        # Extract drift info from snapshot dict
        try:
            report_dict = snapshot.dump_dict()
            for metric_result in report_dict.get("metrics", []):
                metric_data = metric_result.get("result", {})
                if "drift_share" in metric_data:
                    drift_summary["drift_share"] = metric_data["drift_share"]
                    drift_summary["drift_detected"] = metric_data.get("dataset_drift", False)
                    drift_summary["number_of_drifted_columns"] = metric_data.get("number_of_drifted_columns", 0)
                    for col, col_data in metric_data.get("drift_by_columns", {}).items():
                        if col_data.get("drift_detected", False):
                            drift_summary["drifted_columns"].append(col)
        except Exception as e:
            logger.warning(f"Could not extract detailed drift metrics: {e}")

        if drift_summary["drift_detected"]:
            drift_summary["status"] = "drift_detected"
            logger.warning(f"⚠️  DRIFT DETECTED in '{table}': {drift_summary['drifted_columns']}")
        else:
            logger.info(f"✅ No drift detected in '{table}'")

        return drift_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    monitor = DriftMonitor()
    result = monitor.run_drift_check("leads")
    print(json.dumps(result, indent=2, default=str))
