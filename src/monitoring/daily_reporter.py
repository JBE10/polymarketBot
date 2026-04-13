import csv
import logging
from pathlib import Path
from datetime import datetime, timezone

from src.storage.database import Database

log = logging.getLogger(__name__)

class DailyReporter:
    """
    Paper Trading Daily Reporting Mechanism.
    Exports execution logs validating expected vs realized mathematical edge outputs.
    """
    def __init__(self, db: Database, output_dir: str = "data/reports"):
        self.db = db
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    async def build_daily_report(self, date_utc: str | None = None) -> dict:
        report = await self.db.get_daily_decision_report(date_utc=date_utc)
        target_date = report["date_utc"][0]["date"]

        self._write_csv(
            Path(self.output_dir) / f"daily_report_asset_{target_date}.csv",
            report["by_asset"],
        )
        self._write_csv(
            Path(self.output_dir) / f"daily_report_regime_{target_date}.csv",
            report["by_regime"],
        )
        self._write_csv(
            Path(self.output_dir) / f"daily_report_hour_{target_date}.csv",
            report["by_hour"],
        )
        self._write_csv(
            Path(self.output_dir) / f"daily_report_side_{target_date}.csv",
            report["by_side"],
        )

        log.info("Daily decision report exported for %s into %s", target_date, self.output_dir)
        return report

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            with path.open("w", newline="", encoding="utf-8") as f:
                f.write("\n")
            return

        headers = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

