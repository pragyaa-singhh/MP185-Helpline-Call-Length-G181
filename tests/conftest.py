from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "data" / "processed" / "train_clean.csv"


@pytest.fixture(scope="session", autouse=True)
def ensure_clean_data_exists():
    """Ensure minimal clean data exists for CI environment where raw CSVs are omitted."""
    if not CLEAN_DATA.exists():
        CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
        synthetic_df = pd.DataFrame([
            {
                "record_id": f"REC_{i:04d}",
                "date": "2026-03-01",
                "queue_wait_sec": 60.0 + i,
                "agent_exp_months": 12.0,
                "previous_calls": 2,
                "issue_count": 1,
                "hour": 10 + (i % 8),
                "caller_age": 30 + i,
                "language": "english",
                "team": "team alpha",
                "handle_min": 10.0 + (i * 0.5)
            }
            for i in range(10)
        ])
        synthetic_df.to_csv(CLEAN_DATA, index=False)
