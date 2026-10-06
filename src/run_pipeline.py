"""Run cleaning + join + feature engineering and export master tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import build_master_tables, export_outputs, integrity_checks, project_root


def main() -> None:
    print("Building master tables...")
    bundle = build_master_tables()
    out = export_outputs(bundle)

    checks = integrity_checks(bundle["master_train"], bundle["master_test"])
    print("\n=== Integrity checks ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} | {detail}")

    print("\n=== Shapes ===")
    print("master_train:", bundle["master_train"].shape)
    print("master_test:", bundle["master_test"].shape)
    print("cleaning_log rows:", len(bundle["cleaning_log"]))
    print("weather audit:", bundle["weather_audit"])
    print("events audit:", bundle["events_audit"])
    print("zones after:", bundle["zone_after"])

    audit_path = project_root() / "reports" / "A4_join_audit.json"
    audit_path.write_text(
        json.dumps(
            {"weather": bundle["weather_audit"], "events": bundle["events_audit"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle["cleaning_log"].to_csv(project_root() / "reports" / "A1_cleaning_log.csv", index=False)
    print(f"\nExported to {out}")
    print("Reports written under reports/")


if __name__ == "__main__":
    main()
