import os
import pandas as pd
import statistics
from datetime import datetime
from typing import Optional, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def analyze_csv(
    file_path: str,
    question: str,
    category_filter: Optional[str] = None,
    region_filter: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a CSV file and answer a question about it with advanced filtering.
    
    Args:
        file_path: Path to the CSV file.
        question: The analytical question to answer.
        category_filter: Optional category to filter by.
        region_filter: Optional region to filter by.
        date_start: Optional start date (YYYY-MM-DD format).
        date_end: Optional end date (YYYY-MM-DD format).
        
    Returns:
        Dict with comprehensive analysis including summary stats, breakdowns,
        and performance metrics.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If date formats are invalid.
    """
    try:
        # Resolve file path
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        
        # Parse date column if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Apply filters
        if date_start:
            df = df[df["date"] >= pd.to_datetime(date_start)]
        if date_end:
            df = df[df["date"] <= pd.to_datetime(date_end)]
        if category_filter and "category" in df.columns:
            df = df[df["category"] == category_filter]
        if region_filter and "region" in df.columns:
            df = df[df["region"] == region_filter]

        # Build comprehensive summary
        summary = {
            "rows": len(df),
            "columns": list(df.columns),
            "question_analyzed": question,
        }

        # Revenue analysis
        if "revenue" in df.columns:
            summary["total_revenue"] = float(df["revenue"].sum())
            summary["avg_revenue_per_order"] = float(df["revenue"].mean())
            summary["max_order_value"] = float(df["revenue"].max())
            summary["min_order_value"] = float(df["revenue"].min())

        # Category breakdown
        if "category" in df.columns:
            category_stats = df.groupby("category").agg(
                {
                    "revenue": ["sum", "count", "mean"],
                    "quantity": "sum",
                }
            ).round(2)
            summary["revenue_by_category"] = (
                df.groupby("category")["revenue"].sum().sort_values(ascending=False).to_dict()
            )
            summary["orders_by_category"] = (
                df.groupby("category").size().to_dict()
            )

        # Regional performance
        if "region" in df.columns:
            summary["revenue_by_region"] = (
                df.groupby("region")["revenue"].sum().sort_values(ascending=False).to_dict()
            )
            summary["orders_by_region"] = df.groupby("region").size().to_dict()

        # Top products
        if "product" in df.columns and "revenue" in df.columns:
            top_products = (
                df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(5)
            )
            summary["top_5_products"] = top_products.to_dict()

        # Time-based analysis
        if "date" in df.columns and len(df) > 1:
            df_sorted = df.sort_values("date")
            summary["date_range"] = {
                "start": str(df_sorted["date"].min().date()),
                "end": str(df_sorted["date"].max().date()),
            }
            # Monthly trends if applicable
            if len(df_sorted) > 5:
                monthly = df_sorted.set_index("date").resample("M")["revenue"].sum()
                summary["monthly_revenue_trend"] = monthly.to_dict()

        return summary

    except Exception as e:
        return {"error": str(e), "question": question}


def calculate_metrics(data: List[float], metric: str = "mean") -> Dict[str, Any]:
    """Calculate statistical metrics on a list of numbers.
    
    Args:
        data: List of numeric values to analyze.
        metric: One of "mean", "median", "sum", "std_dev", "min", "max",
                "quartiles", "growth_rate", or "all" for comprehensive stats.
                
    Returns:
        Dict with the computed metric(s) and additional stats.
        
    Raises:
        ValueError: If data is empty or metric is not recognized.
        TypeError: If data contains non-numeric values.
    """
    if not data:
        raise ValueError("Data list cannot be empty")

    try:
        # Convert to floats to ensure numeric type
        numeric_data = [float(x) for x in data]
    except (ValueError, TypeError) as e:
        raise TypeError(f"All data values must be numeric: {e}")

    result = {"data_points": len(numeric_data)}

    # Calculate requested metric(s)
    if metric == "all" or metric == "comprehensive":
        result["mean"] = round(statistics.mean(numeric_data), 4)
        result["median"] = round(statistics.median(numeric_data), 4)
        result["sum"] = round(sum(numeric_data), 4)
        result["min"] = round(min(numeric_data), 4)
        result["max"] = round(max(numeric_data), 4)

        if len(numeric_data) > 1:
            result["std_dev"] = round(statistics.stdev(numeric_data), 4)
            result["variance"] = round(statistics.variance(numeric_data), 4)

        # Calculate quartiles
        sorted_data = sorted(numeric_data)
        q1_idx = len(sorted_data) // 4
        q3_idx = (3 * len(sorted_data)) // 4
        result["quartiles"] = {
            "q1": round(sorted_data[q1_idx], 4),
            "median": result["median"],
            "q3": round(sorted_data[q3_idx], 4),
        }

        # Growth rate (change from first to last)
        if len(numeric_data) > 1:
            growth = ((numeric_data[-1] - numeric_data[0]) / numeric_data[0] * 100) if numeric_data[0] != 0 else 0
            result["growth_rate_percent"] = round(growth, 2)

    elif metric == "mean":
        result["value"] = round(statistics.mean(numeric_data), 4)
    elif metric == "median":
        result["value"] = round(statistics.median(numeric_data), 4)
    elif metric == "sum":
        result["value"] = round(sum(numeric_data), 4)
    elif metric == "std_dev":
        if len(numeric_data) < 2:
            raise ValueError("Need at least 2 data points for std_dev")
        result["value"] = round(statistics.stdev(numeric_data), 4)
    elif metric == "min":
        result["value"] = round(min(numeric_data), 4)
    elif metric == "max":
        result["value"] = round(max(numeric_data), 4)
    elif metric == "growth_rate":
        if len(numeric_data) < 2:
            raise ValueError("Need at least 2 data points for growth rate")
        growth = ((numeric_data[-1] - numeric_data[0]) / numeric_data[0] * 100) if numeric_data[0] != 0 else 0
        result["growth_rate_percent"] = round(growth, 2)
    else:
        raise ValueError(
            f"Unknown metric: {metric}. Supported: mean, median, sum, std_dev, min, max, growth_rate, all"
        )

    result["metric"] = metric
    return result


def detect_anomalies(
    file_path: str,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """Detect anomalies/outliers in numeric columns using statistical methods.
    
    Args:
        file_path: Path to the CSV file.
        column: Column name to analyze for anomalies.
        method: Detection method - "iqr" (Interquartile Range) or "zscore".
        threshold: Sensitivity threshold (higher = fewer anomalies detected).
                  For IQR: multiplier for quartile range (1.5 standard).
                  For Z-score: number of standard deviations (3 standard).
        
    Returns:
        Dict with anomaly statistics and flagged records.
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        df = pd.read_csv(file_path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")

        # Remove non-numeric values
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        valid_data = numeric_series.dropna()

        if len(valid_data) < 3:
            raise ValueError("Need at least 3 valid numeric values for anomaly detection")

        result = {
            "column": column,
            "method": method,
            "total_rows": len(df),
            "valid_rows": len(valid_data),
            "missing_rows": len(df) - len(valid_data),
        }

        if method == "iqr":
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)

            anomalies = (valid_data < lower_bound) | (valid_data > upper_bound)
            result["lower_bound"] = round(lower_bound, 4)
            result["upper_bound"] = round(upper_bound, 4)

        elif method == "zscore":
            mean = valid_data.mean()
            std = valid_data.std()
            z_scores = (valid_data - mean).abs() / std
            anomalies = z_scores > threshold
            result["mean"] = round(mean, 4)
            result["std_dev"] = round(std, 4)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

        anomaly_indices = anomalies[anomalies].index.tolist()
        result["anomalies_detected"] = len(anomaly_indices)
        result["anomaly_percentage"] = round((len(anomaly_indices) / len(valid_data)) * 100, 2)

        if len(anomaly_indices) > 0:
            anomaly_values = valid_data[anomaly_indices].tolist()
            result["anomaly_values"] = [round(v, 4) for v in anomaly_values[:10]]
            if len(anomaly_values) > 10:
                result["anomaly_values_truncated"] = f"... and {len(anomaly_values) - 10} more"

        return result

    except Exception as e:
        return {"error": str(e), "column": column}


def compare_segments(
    file_path: str,
    metric_column: str,
    segment_column: str,
    aggregation: str = "sum",
) -> Dict[str, Any]:
    """Compare performance metrics across different segments.
    
    Args:
        file_path: Path to the CSV file.
        metric_column: Column with numeric values to compare (e.g., 'revenue').
        segment_column: Column to segment by (e.g., 'category', 'region').
        aggregation: How to aggregate metrics - "sum", "mean", "count", "max".
        
    Returns:
        Dict with comparative analysis including rankings and differences.
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        df = pd.read_csv(file_path)

        if metric_column not in df.columns or segment_column not in df.columns:
            raise ValueError(f"Columns not found: {metric_column} or {segment_column}")

        # Group and aggregate
        grouped = df.groupby(segment_column)[metric_column].agg(aggregation)
        grouped_sorted = grouped.sort_values(ascending=False)

        result = {
            "metric": metric_column,
            "segment": segment_column,
            "aggregation": aggregation,
            "total_segments": len(grouped_sorted),
            "segment_results": grouped_sorted.round(2).to_dict(),
        }

        # Calculate variance metrics
        result["segment_stats"] = {
            "max": round(grouped_sorted.max(), 4),
            "min": round(grouped_sorted.min(), 4),
            "mean": round(grouped_sorted.mean(), 4),
            "std_dev": round(grouped_sorted.std(), 4) if len(grouped_sorted) > 1 else 0,
        }

        # Top and bottom performers
        result["top_performer"] = {
            "segment": grouped_sorted.index[0],
            "value": round(grouped_sorted.iloc[0], 4),
        }
        result["bottom_performer"] = {
            "segment": grouped_sorted.index[-1],
            "value": round(grouped_sorted.iloc[-1], 4),
        }

        # Performance gap
        gap = grouped_sorted.iloc[0] - grouped_sorted.iloc[-1]
        gap_pct = (gap / grouped_sorted.iloc[-1] * 100) if grouped_sorted.iloc[-1] != 0 else 0
        result["performance_gap"] = {
            "absolute": round(gap, 4),
            "percentage": round(gap_pct, 2),
        }

        return result

    except Exception as e:
        return {"error": str(e), "metric_column": metric_column, "segment_column": segment_column}


def get_data_quality_report(file_path: str) -> Dict[str, Any]:
    """Generate a comprehensive data quality report.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        Dict with data quality metrics, missing values, duplicates, and type info.
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        df = pd.read_csv(file_path)

        report = {
            "file": os.path.basename(file_path),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        }

        # Missing values
        missing_summary = df.isnull().sum()
        report["missing_values"] = {
            col: int(count) for col, count in missing_summary[missing_summary > 0].items()
        }
        report["completeness_percentage"] = round(
            ((len(df) * len(df.columns) - missing_summary.sum()) / (len(df) * len(df.columns))) * 100, 2
        )

        # Duplicates
        duplicate_count = df.duplicated().sum()
        report["duplicate_rows"] = int(duplicate_count)
        report["duplicate_percentage"] = round((duplicate_count / len(df)) * 100, 2)

        # Column information
        report["columns"] = []
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
            }

            if df[col].dtype in ["int64", "float64"]:
                col_info.update({
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                })

            report["columns"].append(col_info)

        # Quality score
        completeness = report["completeness_percentage"]
        uniqueness = 100 - report["duplicate_percentage"]
        quality_score = (completeness + uniqueness) / 2
        report["overall_quality_score"] = round(quality_score, 2)
        report["quality_grade"] = (
            "Excellent" if quality_score >= 95
            else "Good" if quality_score >= 85
            else "Fair" if quality_score >= 70
            else "Poor"
        )

        return report

    except Exception as e:
        return {"error": str(e)}


def generate_insights(file_path: str, question: str = "") -> Dict[str, Any]:
    """Automatically generate key insights and findings from data.
    
    Args:
        file_path: Path to the CSV file.
        question: Optional specific question to focus on.
        
    Returns:
        Dict with automatic insights including trends, patterns, and highlights.
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        df = pd.read_csv(file_path)
        insights = {
            "question": question or "General analysis",
            "data_shape": {"rows": len(df), "columns": len(df.columns)},
            "findings": [],
        }

        # Revenue insights
        if "revenue" in df.columns:
            revenue_stats = {
                "total_revenue": float(df["revenue"].sum()),
                "avg_order_value": float(df["revenue"].mean()),
                "revenue_range": {
                    "min": float(df["revenue"].min()),
                    "max": float(df["revenue"].max()),
                },
            }
            insights["revenue_stats"] = revenue_stats
            insights["findings"].append(f"Total Revenue: ${revenue_stats['total_revenue']:,.2f}")
            insights["findings"].append(f"Average Order Value: ${revenue_stats['avg_order_value']:,.2f}")

        # Category insights
        if "category" in df.columns and "revenue" in df.columns:
            top_category = df.groupby("category")["revenue"].sum().idxmax()
            top_revenue = df.groupby("category")["revenue"].sum().max()
            insights["findings"].append(f"Top Category: {top_category} (${top_revenue:,.2f})")

        # Regional insights
        if "region" in df.columns and "revenue" in df.columns:
            regional_revenue = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
            top_region = regional_revenue.index[0]
            insights["findings"].append(f"Top Region: {top_region} (${regional_revenue.iloc[0]:,.2f})")

        # Product insights
        if "product" in df.columns and "quantity" in df.columns:
            best_seller = df.groupby("product")["quantity"].sum().idxmax()
            best_quantity = df.groupby("product")["quantity"].sum().max()
            insights["findings"].append(f"Best-Selling Product: {best_seller} ({int(best_quantity)} units)")

        # Time-based insights
        if "date" in df.columns:
            df_sorted = df.copy()
            df_sorted["date"] = pd.to_datetime(df_sorted["date"])
            date_range = (df_sorted["date"].max() - df_sorted["date"].min()).days
            insights["findings"].append(f"Data Span: {date_range} days")

        # Distribution insights
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            skewness = (df[col].mean() - df[col].median()) / df[col].std() if df[col].std() > 0 else 0
            insights["findings"].append(
                f"Data Distribution: {'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'}"
            )

        return insights

    except Exception as e:
        return {"error": str(e), "question": question}