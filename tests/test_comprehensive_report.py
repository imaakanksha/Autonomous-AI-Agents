import pandas as pd

from data_analyst_agent.tools import generate_comprehensive_report


def test_generate_comprehensive_report_returns_structured_summary(tmp_path):
    csv_path = tmp_path / "sales.csv"
    pd.DataFrame(
        [
            {"date": "2024-01-01", "category": "A", "region": "North", "product": "P1", "revenue": 100, "quantity": 10, "score": 1.0},
            {"date": "2024-01-02", "category": "A", "region": "South", "product": "P2", "revenue": 120, "quantity": 12, "score": 1.2},
            {"date": "2024-01-03", "category": "B", "region": "North", "product": "P1", "revenue": 80, "quantity": 8, "score": 0.9},
            {"date": "2024-01-04", "category": "B", "region": "South", "product": "P3", "revenue": 70, "quantity": 7, "score": 0.8},
            {"date": "2024-01-05", "category": "C", "region": "North", "product": "P3", "revenue": 300, "quantity": 30, "score": 2.8},
        ]
    ).to_csv(csv_path, index=False)

    report = generate_comprehensive_report(
        str(csv_path),
        question="What drives revenue?",
        target_column="revenue",
        top_n=2,
    )

    assert "question" in report
    assert "executive_summary" in report
    assert "segment_comparisons" in report
    assert "anomaly_detection" in report
    assert "key_drivers" in report
    assert report["quality_grade"] in {"Excellent", "Good", "Fair", "Poor"}
