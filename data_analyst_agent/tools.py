import os
import pandas as pd
import numpy as np
import statistics
from datetime import datetime
from typing import Optional, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(file_path: str) -> str:
    if not os.path.isabs(file_path):
        file_path = os.path.join(BASE_DIR, file_path)
    return os.path.normpath(file_path)


def load_csv(file_path: str) -> pd.DataFrame:
    file_path = resolve_path(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path)


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    if "date" in df.columns:
        return "date"
    candidates = [col for col in df.columns if "date" in col.lower()]
    return candidates[0] if candidates else None


def parse_optional_date(value: str) -> datetime:
    try:
        return pd.to_datetime(value)
    except Exception as e:
        raise ValueError(f"Unable to parse date '{value}': {e}")


def safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def analyze_csv(
    file_path: str,
    question: str,
    category_filter: Optional[str] = None,
    region_filter: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    top_n: int = 5,
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
        df = load_csv(file_path)

        date_column = "date" if "date" in df.columns else detect_date_column(df)
        if date_column is not None:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        if date_start:
            if not date_column:
                raise ValueError("Date filter requires a date column in the dataset")
            df = df[df[date_column] >= parse_optional_date(date_start)]
        if date_end:
            if not date_column:
                raise ValueError("Date filter requires a date column in the dataset")
            df = df[df[date_column] <= parse_optional_date(date_end)]
        if category_filter and "category" in df.columns:
            df = df[df["category"] == category_filter]
        if region_filter and "region" in df.columns:
            df = df[df["region"] == region_filter]

        missing_summary = df.isnull().sum()
        duplicate_count = int(df.duplicated().sum())

        summary = {
            "rows": len(df),
            "columns": list(df.columns),
            "question_analyzed": question,
            "filtered_rows": len(df),
            "missing_values": {col: int(count) for col, count in missing_summary.items() if count > 0},
            "duplicate_rows": duplicate_count,
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        if "revenue" in df.columns:
            revenue = pd.to_numeric(df["revenue"], errors="coerce")
            summary["total_revenue"] = float(revenue.sum())
            summary["avg_revenue_per_order"] = float(revenue.mean())
            summary["max_order_value"] = float(revenue.max())
            summary["min_order_value"] = float(revenue.min())
            summary["revenue_valid_rows"] = int(revenue.notna().sum())
            summary["revenue_missing_rows"] = int(revenue.isna().sum())

        if "category" in df.columns and "revenue" in df.columns:
            revenue_by_category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
            summary["revenue_by_category"] = revenue_by_category.round(2).to_dict()
            summary["orders_by_category"] = df.groupby("category").size().to_dict()
            summary["top_categories"] = [
                {"category": idx, "total_revenue": float(val)}
                for idx, val in revenue_by_category.head(top_n).items()
            ]

        if "region" in df.columns and "revenue" in df.columns:
            revenue_by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
            summary["revenue_by_region"] = revenue_by_region.round(2).to_dict()
            summary["orders_by_region"] = df.groupby("region").size().to_dict()
            summary["top_regions"] = [
                {"region": idx, "total_revenue": float(val)}
                for idx, val in revenue_by_region.head(top_n).items()
            ]

        if "product" in df.columns and "revenue" in df.columns:
            top_products = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(top_n)
            summary["top_products"] = [
                {"product": idx, "revenue": float(val)}
                for idx, val in top_products.items()
            ]

        if date_column is not None and len(df) > 1:
            df_sorted = df.sort_values(date_column)
            summary["date_range"] = {
                "start": str(df_sorted[date_column].min().date()),
                "end": str(df_sorted[date_column].max().date()),
            }
            if len(df_sorted) > 5 and "revenue" in df.columns:
                monthly = df_sorted.set_index(date_column).resample("M")["revenue"].sum()
                summary["monthly_revenue_trend"] = [
                    {"period": str(index.date()), "revenue": float(value)}
                    for index, value in monthly.items()
                ]
                change = monthly.pct_change().dropna()
                if not change.empty:
                    summary["revenue_growth_rate_percent"] = round(float(change.iloc[-1] * 100), 2)

        summary["top_takeaways"] = []
        if "revenue_by_category" in summary:
            top_category = next(iter(summary["revenue_by_category"]))
            summary["top_takeaways"].append(f"Top category by revenue: {top_category}")
        if "revenue_by_region" in summary:
            top_region = next(iter(summary["revenue_by_region"]))
            summary["top_takeaways"].append(f"Top region by revenue: {top_region}")
        if "top_products" in summary:
            summary["top_takeaways"].append(
                f"Top product: {summary['top_products'][0]['product']}"
            )

        summary["recommendations"] = []
        if duplicate_count > 0:
            summary["recommendations"].append(
                "Review and remove duplicate records to ensure reliable aggregate results."
            )
        if summary["missing_values"]:
            summary["recommendations"].append(
                "Address missing values in key fields to improve data completeness."
            )
        if "revenue_growth_rate_percent" in summary:
            direction = "upward" if summary["revenue_growth_rate_percent"] > 0 else "downward" if summary["revenue_growth_rate_percent"] < 0 else "stable"
            summary["recommendations"].append(
                f"Current monthly revenue trend is {direction}."
            )

        return summary

    except Exception as e:
        return {"error": str(e), "question": question}




def identify_key_drivers(file_path: str, target_column: str, top_n: int = 3) -> Dict[str, Any]:
    """Identify the most predictive numeric features for a target column.

    Args:
        file_path: Path to the CSV file.
        target_column: Numeric target column to explain.
        top_n: Number of top drivers to return.

    Returns:
        Dict with correlations and ranked drivers.
    """
    try:
        df = load_csv(file_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' must be numeric")

        driver_cols = [col for col in numeric_df.columns if col != target_column]
        if not driver_cols:
            raise ValueError("No other numeric columns available to compare against the target")

        correlations = numeric_df[driver_cols].corrwith(numeric_df[target_column]).dropna()
        ranked = correlations.abs().sort_values(ascending=False)

        result = {
            "target": target_column,
            "driver_count": len(ranked),
            "top_drivers": [],
            "correlations": correlations.round(4).to_dict(),
        }

        for idx, value in ranked.head(top_n).items():
            result["top_drivers"].append(
                {
                    "feature": idx,
                    "correlation": round(float(correlations[idx]), 4),
                    "absolute_correlation": round(float(value), 4),
                }
            )

        return result
    except Exception as e:
        return {"error": str(e), "target_column": target_column}


def calculate_metrics(data: List[float], metric: str = "mean") -> Dict[str, Any]:
    """Calculate statistical metrics on a list of numbers.
    
    Args:
        data: List of numeric values to analyze.
        metric: One of "mean", "median", "sum", "std_dev", "variance", "min",
                "max", "growth_rate", "percent_change", "quartiles", "all", or
                "comprehensive" for full statistics.
                
    Returns:
        Dict with the computed metric(s) and additional stats.
        
    Raises:
        ValueError: If data is empty or metric is not recognized.
        TypeError: If data contains non-numeric values.
    """
    if not data:
        raise ValueError("Data list cannot be empty")

    try:
        numeric_data = [float(x) for x in data]
    except (ValueError, TypeError) as e:
        raise TypeError(f"All data values must be numeric: {e}")

    result = {"data_points": len(numeric_data)}

    if metric in {"all", "comprehensive"}:
        result["mean"] = round(statistics.mean(numeric_data), 4)
        result["median"] = round(statistics.median(numeric_data), 4)
        result["sum"] = round(sum(numeric_data), 4)
        result["min"] = round(min(numeric_data), 4)
        result["max"] = round(max(numeric_data), 4)

        if len(numeric_data) > 1:
            result["std_dev"] = round(statistics.stdev(numeric_data), 4)
            result["variance"] = round(statistics.variance(numeric_data), 4)

        sorted_data = sorted(numeric_data)
        q1_idx = max(0, len(sorted_data) // 4)
        q3_idx = min(len(sorted_data) - 1, (3 * len(sorted_data)) // 4)
        result["quartiles"] = {
            "q1": round(sorted_data[q1_idx], 4),
            "median": result["median"],
            "q3": round(sorted_data[q3_idx], 4),
        }

        if len(numeric_data) > 1:
            growth = (
                (numeric_data[-1] - numeric_data[0]) / numeric_data[0] * 100
            ) if numeric_data[0] != 0 else 0
            result["growth_rate_percent"] = round(growth, 2)
            result["percent_change"] = result["growth_rate_percent"]

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
    elif metric == "variance":
        if len(numeric_data) < 2:
            raise ValueError("Need at least 2 data points for variance")
        result["value"] = round(statistics.variance(numeric_data), 4)
    elif metric == "min":
        result["value"] = round(min(numeric_data), 4)
    elif metric == "max":
        result["value"] = round(max(numeric_data), 4)
    elif metric in {"growth_rate", "percent_change"}:
        if len(numeric_data) < 2:
            raise ValueError("Need at least 2 data points for growth rate")
        change = (
            (numeric_data[-1] - numeric_data[0]) / numeric_data[0] * 100
        ) if numeric_data[0] != 0 else 0
        result["growth_rate_percent"] = round(change, 2)
        if metric == "percent_change":
            result["value"] = result["growth_rate_percent"]
    elif metric == "quartiles":
        sorted_data = sorted(numeric_data)
        q1_idx = max(0, len(sorted_data) // 4)
        q3_idx = min(len(sorted_data) - 1, (3 * len(sorted_data)) // 4)
        result["value"] = {
            "q1": round(sorted_data[q1_idx], 4),
            "median": round(statistics.median(sorted_data), 4),
            "q3": round(sorted_data[q3_idx], 4),
        }
    else:
        raise ValueError(
            f"Unknown metric: {metric}. Supported: mean, median, sum, std_dev, variance, min, max, growth_rate, percent_change, quartiles, all, comprehensive"
        )

    result["metric"] = metric
    return result


def detect_anomalies(
    file_path: str,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
    return_rows: bool = False,
    context_columns: Optional[List[str]] = None,
    max_records: int = 10,
) -> Dict[str, Any]:
    """Detect anomalies/outliers in numeric columns using statistical methods.
    
    Args:
        file_path: Path to the CSV file.
        column: Column name to analyze for anomalies.
        method: Detection method - "iqr" (Interquartile Range) or "zscore".
        threshold: Sensitivity threshold (higher = fewer anomalies detected).
                  For IQR: multiplier for quartile range (1.5 standard).
                  For Z-score: number of standard deviations (3 standard).
        return_rows: If true, include sample rows for detected anomalies.
        context_columns: Optional additional columns to include with anomaly rows.
        max_records: Maximum sample anomaly rows to return.
        
    Returns:
        Dict with anomaly statistics and flagged records.
    """
    try:
        df = load_csv(file_path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")

        numeric_series = safe_numeric_series(df[column])
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
            if std == 0:
                raise ValueError("Z-score anomaly detection requires variation in the data")
            z_scores = (valid_data - mean).abs() / std
            anomalies = z_scores > threshold
            result["mean"] = round(mean, 4)
            result["std_dev"] = round(std, 4)
            result["detection_methods"] = ["zscore"]

        elif method == "both":
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            mean = valid_data.mean()
            std = valid_data.std()
            if std == 0:
                raise ValueError("Combined anomaly detection requires variation in the data")
            z_scores = (valid_data - mean).abs() / std
            anomalies = ((valid_data < lower_bound) | (valid_data > upper_bound)) | (z_scores > threshold)
            result["lower_bound"] = round(lower_bound, 4)
            result["upper_bound"] = round(upper_bound, 4)
            result["mean"] = round(mean, 4)
            result["std_dev"] = round(std, 4)
            result["detection_methods"] = ["iqr", "zscore"]

        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr', 'zscore', or 'both'")

        anomaly_indices = anomalies[anomalies].index.tolist()
        result["anomalies_detected"] = len(anomaly_indices)
        result["anomaly_percentage"] = round((len(anomaly_indices) / len(valid_data)) * 100, 2)

        if anomaly_indices:
            anomaly_values = valid_data.loc[anomaly_indices].tolist()
            result["anomaly_values"] = [round(float(v), 4) for v in anomaly_values[:10]]
            if len(anomaly_values) > 10:
                result["anomaly_values_truncated"] = f"... and {len(anomaly_values) - 10} more"

            if return_rows:
                context_columns = [column] + [col for col in (context_columns or []) if col in df.columns and col != column]
                sample_rows = df.loc[anomaly_indices, context_columns].head(max_records)
                result["anomaly_rows"] = sample_rows.to_dict(orient="records")

        return result

    except Exception as e:
        return {"error": str(e), "column": column}


def compare_segments(
    file_path: str,
    metric_column: str,
    segment_column: str,
    aggregation: str = "sum",
    top_n: int = 3,
    sort_ascending: bool = False,
) -> Dict[str, Any]:
    """Compare performance metrics across different segments.
    
    Args:
        file_path: Path to the CSV file.
        metric_column: Column with numeric values to compare (e.g., 'revenue').
        segment_column: Column to segment by (e.g., 'category', 'region').
        aggregation: How to aggregate metrics - "sum", "mean", "count", "max", "min", "median".
        top_n: Number of top/bottom segments to return.
        sort_ascending: When true, rank ascending rather than descending.
        
    Returns:
        Dict with comparative analysis including rankings and differences.
    """
    try:
        df = load_csv(file_path)

        if metric_column not in df.columns or segment_column not in df.columns:
            raise ValueError(f"Columns not found: {metric_column} or {segment_column}")

        if aggregation in {"sum", "mean", "min", "max", "median"}:
            df[metric_column] = safe_numeric_series(df[metric_column])

        grouped = df.groupby(segment_column)[metric_column].agg(aggregation).dropna()
        grouped_sorted = grouped.sort_values(ascending=sort_ascending)

        result = {
            "metric": metric_column,
            "segment": segment_column,
            "aggregation": aggregation,
            "total_segments": len(grouped_sorted),
            "segment_results": grouped_sorted.round(2).to_dict(),
            "top_segments": [
                {"segment": idx, "value": float(val)}
                for idx, val in grouped_sorted.head(top_n).items()
            ],
            "bottom_segments": [
                {"segment": idx, "value": float(val)}
                for idx, val in grouped_sorted.tail(top_n).items()
            ],
        }

        if not grouped_sorted.empty:
            result["segment_stats"] = {
                "max": round(grouped_sorted.max(), 4),
                "min": round(grouped_sorted.min(), 4),
                "mean": round(grouped_sorted.mean(), 4),
                "std_dev": round(grouped_sorted.std(), 4) if len(grouped_sorted) > 1 else 0,
            }
            result["top_performer"] = {
                "segment": grouped_sorted.index[0],
                "value": round(float(grouped_sorted.iloc[0]), 4),
            }
            result["bottom_performer"] = {
                "segment": grouped_sorted.index[-1],
                "value": round(float(grouped_sorted.iloc[-1]), 4),
            }
            gap = float(grouped_sorted.iloc[0]) - float(grouped_sorted.iloc[-1])
            gap_pct = (gap / float(grouped_sorted.iloc[-1]) * 100) if float(grouped_sorted.iloc[-1]) != 0 else 0
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
        df = load_csv(file_path)

        total_cells = len(df) * len(df.columns)
        missing_summary = df.isnull().sum()
        duplicate_count = int(df.duplicated().sum())

        report = {
            "file": os.path.basename(resolve_path(file_path)),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
            "missing_values": {col: int(count) for col, count in missing_summary[missing_summary > 0].items()},
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0,
            "completeness_percentage": round(
                ((total_cells - missing_summary.sum()) / total_cells) * 100, 2
            ) if total_cells else 100,
        }

        report["high_missing_columns"] = [
            col for col, count in missing_summary.items() if len(df) and (count / len(df)) >= 0.2
        ]
        report["inconsistent_type_columns"] = []

        report["columns"] = []
        for col in df.columns:
            col_series = df[col]
            non_null = int(col_series.notna().sum())
            null_count = int(col_series.isna().sum())
            col_info = {
                "name": col,
                "type": str(col_series.dtype),
                "non_null_count": non_null,
                "null_count": null_count,
                "unique_values": int(col_series.nunique(dropna=True)),
            }

            if str(col_series.dtype) in ["int64", "float64"]:
                numeric = col_series.dropna().astype(float)
                if not numeric.empty:
                    col_info.update({
                        "min": float(numeric.min()),
                        "max": float(numeric.max()),
                        "mean": float(numeric.mean()),
                        "std_dev": float(numeric.std()),
                    })
            elif str(col_series.dtype) == "object":
                empty_strings = int((col_series.astype(str).str.strip() == "").sum())
                col_info["empty_string_count"] = empty_strings
                parsed_numeric = pd.to_numeric(col_series, errors="coerce")
                if parsed_numeric.notna().any() and parsed_numeric.isna().any():
                    report["inconsistent_type_columns"].append(col)

            report["columns"].append(col_info)

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
        df = load_csv(file_path)
        insights = {
            "question": question or "General analysis",
            "data_shape": {"rows": len(df), "columns": len(df.columns)},
            "findings": [],
            "warnings": [],
        }

        quality_report = get_data_quality_report(file_path)
        if isinstance(quality_report, dict) and quality_report.get("completeness_percentage", 100) < 90:
            insights["warnings"].append(
                "Dataset completeness is below 90%; review missing values before drawing firm conclusions."
            )
        if isinstance(quality_report, dict) and quality_report.get("duplicate_percentage", 0) > 10:
            insights["warnings"].append("High duplicate row rate; duplicates may distort aggregate metrics.")

        if "revenue" in df.columns:
            revenue = pd.to_numeric(df["revenue"], errors="coerce")
            insights["revenue_stats"] = {
                "total_revenue": float(revenue.sum()),
                "avg_order_value": float(revenue.mean()),
                "revenue_range": {"min": float(revenue.min()), "max": float(revenue.max())},
            }
            insights["findings"].append(f"Total revenue is ${insights['revenue_stats']['total_revenue']:,.2f}.")
            insights["findings"].append(f"Average order value is ${insights['revenue_stats']['avg_order_value']:,.2f}.")

        if "category" in df.columns and "revenue" in df.columns:
            revenue_by_category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
            if not revenue_by_category.empty:
                top_category = revenue_by_category.index[0]
                top_revenue = revenue_by_category.iloc[0]
                insights["findings"].append(
                    f"Top revenue-generating category: {top_category} with ${top_revenue:,.2f}."
                )

        if "region" in df.columns and "revenue" in df.columns:
            revenue_by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
            if not revenue_by_region.empty:
                top_region = revenue_by_region.index[0]
                insights["findings"].append(
                    f"Top revenue region: {top_region} with ${revenue_by_region.iloc[0]:,.2f}."
                )

        if "product" in df.columns and "quantity" in df.columns:
            sales_by_product = df.groupby("product")["quantity"].sum().sort_values(ascending=False)
            if not sales_by_product.empty:
                best_seller = sales_by_product.index[0]
                best_quantity = sales_by_product.iloc[0]
                insights["findings"].append(
                    f"Best-selling product: {best_seller} ({int(best_quantity)} units sold)."
                )
                insights["top_products"] = [
                    {"product": idx, "quantity": int(val)}
                    for idx, val in sales_by_product.head(3).items()
                ]

        date_column = "date" if "date" in df.columns else detect_date_column(df)
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            valid_dates = df[date_column].dropna()
            if len(valid_dates) > 1:
                date_range = (valid_dates.max() - valid_dates.min()).days
                insights["findings"].append(f"Data covers {date_range} days.")
                if "revenue" in df.columns:
                    trend = (
                        df.set_index(date_column).resample("M")["revenue"].sum().pct_change().dropna()
                    )
                    if not trend.empty:
                        latest = trend.iloc[-1] * 100
                        direction = "increasing" if latest > 0 else "decreasing" if latest < 0 else "flat"
                        insights["findings"].append(
                            f"Most recent monthly revenue trend is {direction} ({latest:.2f}%)."
                        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            pairs = (
                corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
            )
            if not pairs.empty:
                strongest = pairs.head(3)
                insights["top_correlations"] = [
                    {"pair": f"{idx[0]} & {idx[1]}", "correlation": round(float(value), 4)}
                    for idx, value in strongest.items()
                ]
                for idx, value in strongest.items():
                    insights["findings"].append(
                        f"Strong relationship between {idx[0]} and {idx[1]} (correlation {value:.2f})."
                    )

        if "revenue" in df.columns and "product" in df.columns:
            revenue_by_product = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
            if len(revenue_by_product) > 1:
                top_share = revenue_by_product.head(1).sum() / revenue_by_product.sum() * 100
                insights["findings"].append(
                    f"Top product contributes {top_share:.2f}% of total revenue."
                )

        if not insights["findings"]:
            insights["findings"].append("No strong insights could be generated from the current dataset structure.")

        return insights

    except Exception as e:
        return {"error": str(e), "question": question}


def generate_comprehensive_report(
    file_path: str,
    question: str = "",
    target_column: str = "revenue",
    top_n: int = 3,
) -> Dict[str, Any]:
    """Generate a comprehensive analytics report combining multiple analyses.

    The report combines data quality, segment comparison, anomaly detection,
    key-driver analysis, and a concise executive summary in a single payload.
    """
    try:
        df = load_csv(file_path)
        quality_report = get_data_quality_report(file_path)
        insights = generate_insights(file_path, question=question)
        segment_comparison = compare_segments(
            file_path,
            metric_column=target_column if target_column in df.columns else "revenue",
            segment_column="category" if "category" in df.columns else df.columns[0],
            aggregation="sum",
            top_n=top_n,
        )
        anomaly_detection = detect_anomalies(
            file_path,
            column=target_column if target_column in df.columns else "revenue",
            method="iqr",
            threshold=1.5,
            return_rows=True,
            max_records=5,
        )
        key_drivers = identify_key_drivers(
            file_path,
            target_column=target_column if target_column in df.columns else "revenue",
            top_n=top_n,
        )

        executive_summary = []
        if isinstance(insights, dict) and insights.get("findings"):
            executive_summary.extend(insights["findings"][:3])
        if isinstance(segment_comparison, dict) and segment_comparison.get("top_performer"):
            executive_summary.append(
                f"Best-performing segment is {segment_comparison['top_performer']['segment']} with {segment_comparison['top_performer']['value']}."
            )
        if isinstance(anomaly_detection, dict) and anomaly_detection.get("anomalies_detected"):
            executive_summary.append(
                f"Detected {anomaly_detection['anomalies_detected']} anomalies in {anomaly_detection['column']}."
            )
        if isinstance(key_drivers, dict) and key_drivers.get("top_drivers"):
            driver_names = ", ".join(item["feature"] for item in key_drivers["top_drivers"][:3])
            executive_summary.append(f"Top drivers identified: {driver_names}.")

        report = {
            "question": question or "General analysis",
            "executive_summary": executive_summary,
            "data_shape": {"rows": len(df), "columns": len(df.columns)},
            "quality_report": quality_report,
            "segment_comparisons": segment_comparison,
            "anomaly_detection": anomaly_detection,
            "key_drivers": key_drivers,
            "insights": insights,
            "quality_grade": quality_report.get("quality_grade", "Poor") if isinstance(quality_report, dict) else "Poor",
        }
        return report
    except Exception as e:
        return {"error": str(e), "question": question}


# ==============================================================================
# PREDICTIVE ANALYTICS & FORECASTING
# ==============================================================================

def forecast_revenue_trend(
    file_path: str,
    column: str = "revenue",
    periods_ahead: int = 12,
    date_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Forecast revenue/metrics using linear regression on time-series data.
    
    Args:
        file_path: Path to the CSV file.
        column: Column to forecast (e.g., 'revenue', 'quantity').
        periods_ahead: Number of periods to forecast into the future.
        date_column: Column name containing dates; auto-detected if not provided.
        
    Returns:
        Dict with forecast results, trend line, and confidence bounds.
    """
    try:
        df = load_csv(file_path)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")
        
        # Detect date column
        if date_column is None:
            date_column = detect_date_column(df)
        
        if date_column is None:
            raise ValueError("No date column found; unable to perform time-series forecast")
        
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df[column] = safe_numeric_series(df[column])
        
        # Aggregate by date (daily aggregation)
        ts_data = df.groupby(date_column)[column].sum().sort_index()
        
        if len(ts_data) < 3:
            raise ValueError("Need at least 3 time-series data points for forecasting")
        
        # Create time index
        X = np.arange(len(ts_data)).reshape(-1, 1)
        y = ts_data.values
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast
        future_x = np.arange(len(ts_data), len(ts_data) + periods_ahead).reshape(-1, 1)
        forecast_values = model.predict(future_x)
        
        # Calculate R-squared and slope
        from sklearn.metrics import r2_score
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        # Estimate confidence interval (±15% of recent average)
        recent_avg = np.mean(y[-3:]) if len(y) >= 3 else np.mean(y)
        confidence_margin = recent_avg * 0.15
        
        result = {
            "column": column,
            "historical_periods": len(ts_data),
            "forecast_periods": periods_ahead,
            "trend_slope": round(float(model.coef_[0]), 4),
            "r_squared": round(r_squared, 4),
            "model_fit_quality": "Strong" if r_squared > 0.7 else "Moderate" if r_squared > 0.4 else "Weak",
            "forecast": [
                {
                    "period": i + 1,
                    "value": round(float(val), 2),
                    "confidence_lower": round(float(val - confidence_margin), 2),
                    "confidence_upper": round(float(val + confidence_margin), 2),
                }
                for i, val in enumerate(forecast_values)
            ],
            "recent_average": round(recent_avg, 2),
            "forecast_trend": "increasing" if model.coef_[0] > 0 else "decreasing" if model.coef_[0] < 0 else "stable",
        }
        
        return result
    except ImportError:
        return {"error": "scikit-learn not installed; install with: pip install scikit-learn"}
    except Exception as e:
        return {"error": str(e), "column": column}


def detect_seasonality(
    file_path: str,
    column: str = "revenue",
    date_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect seasonal patterns in time-series data.
    
    Args:
        file_path: Path to the CSV file.
        column: Column to analyze for seasonality.
        date_column: Column name containing dates; auto-detected if not provided.
        
    Returns:
        Dict with seasonal pattern analysis and recommendations.
    """
    try:
        df = load_csv(file_path)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if date_column is None:
            date_column = detect_date_column(df)
        
        if date_column is None:
            raise ValueError("No date column found for seasonality detection")
        
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df[column] = safe_numeric_series(df[column])
        
        ts_data = df.groupby(date_column)[column].sum().sort_index()
        
        # Check for monthly seasonality
        df_temp = df.copy()
        df_temp['month'] = df_temp[date_column].dt.month
        monthly_avg = df_temp.groupby('month')[column].mean()
        monthly_std = monthly_avg.std()
        monthly_coeff_var = (monthly_std / monthly_avg.mean()) * 100 if monthly_avg.mean() != 0 else 0
        
        # Check for day-of-week seasonality if available
        df_temp['dayofweek'] = df_temp[date_column].dt.dayofweek
        weekly_avg = df_temp.groupby('dayofweek')[column].mean()
        weekly_std = weekly_avg.std()
        weekly_coeff_var = (weekly_std / weekly_avg.mean()) * 100 if weekly_avg.mean() != 0 else 0
        
        result = {
            "column": column,
            "data_points": len(ts_data),
            "monthly_seasonality": {
                "detected": monthly_coeff_var > 15,  # >15% coefficient of variation indicates seasonality
                "coefficient_of_variation": round(monthly_coeff_var, 2),
                "top_performing_month": int(monthly_avg.idxmax()) if len(monthly_avg) > 0 else None,
                "monthly_pattern": {int(idx): round(float(val), 2) for idx, val in monthly_avg.items()},
            },
            "weekly_seasonality": {
                "detected": weekly_coeff_var > 15,
                "coefficient_of_variation": round(weekly_coeff_var, 2),
                "best_day": int(weekly_avg.idxmax()) if len(weekly_avg) > 0 else None,
                "day_pattern": {
                    int(idx): round(float(val), 2) 
                    for idx, val in weekly_avg.items()
                },
                "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            },
        }
        
        return result
    except Exception as e:
        return {"error": str(e), "column": column}


# ==============================================================================
# STATISTICAL HYPOTHESIS TESTING & VALIDATION
# ==============================================================================

def statistical_significance_test(
    file_path: str,
    column_a: str,
    column_b: str,
    test_type: str = "ttest",
) -> Dict[str, Any]:
    """Perform statistical significance testing between two columns.
    
    Args:
        file_path: Path to the CSV file.
        column_a: First column to compare.
        column_b: Second column to compare.
        test_type: "ttest" (independent t-test), "mannwhitney" (non-parametric), or "pearson" (correlation).
        
    Returns:
        Dict with test results, p-value, and interpretation.
    """
    try:
        from scipy import stats
        
        df = load_csv(file_path)
        
        if column_a not in df.columns or column_b not in df.columns:
            raise ValueError(f"Columns not found: {column_a} or {column_b}")
        
        data_a = safe_numeric_series(df[column_a]).dropna()
        data_b = safe_numeric_series(df[column_b]).dropna()
        
        if len(data_a) < 2 or len(data_b) < 2:
            raise ValueError("Each column needs at least 2 non-null values for testing")
        
        result = {
            "column_a": column_a,
            "column_b": column_b,
            "test_type": test_type,
            "n_a": len(data_a),
            "n_b": len(data_b),
            "mean_a": round(float(data_a.mean()), 4),
            "mean_b": round(float(data_b.mean()), 4),
            "std_a": round(float(data_a.std()), 4),
            "std_b": round(float(data_b.std()), 4),
        }
        
        if test_type == "ttest":
            t_stat, p_value = stats.ttest_ind(data_a, data_b)
            result["test_statistic"] = round(t_stat, 4)
            result["p_value"] = round(p_value, 6)
            result["significance_level"] = 0.05
            result["is_significant"] = p_value < 0.05
            result["interpretation"] = (
                f"Columns are significantly different (p={p_value:.6f} < 0.05)" 
                if p_value < 0.05 
                else f"No significant difference (p={p_value:.6f} >= 0.05)"
            )
        
        elif test_type == "mannwhitney":
            u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
            result["test_statistic"] = round(u_stat, 4)
            result["p_value"] = round(p_value, 6)
            result["significance_level"] = 0.05
            result["is_significant"] = p_value < 0.05
            result["interpretation"] = (
                f"Distributions are significantly different (p={p_value:.6f} < 0.05)"
                if p_value < 0.05
                else f"No significant difference (p={p_value:.6f} >= 0.05)"
            )
        
        elif test_type == "pearson":
            corr, p_value = stats.pearsonr(data_a, data_b)
            result["correlation"] = round(corr, 4)
            result["p_value"] = round(p_value, 6)
            result["is_significant"] = p_value < 0.05
            result["interpretation"] = (
                f"Significant correlation ({corr:.4f}, p={p_value:.6f})"
                if p_value < 0.05
                else f"No significant correlation (p={p_value:.6f})"
            )
        
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        return result
    
    except ImportError:
        return {"error": "scipy not installed; install with: pip install scipy"}
    except Exception as e:
        return {"error": str(e), "column_a": column_a, "column_b": column_b}


def confidence_interval_analysis(
    file_path: str,
    column: str,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Calculate confidence interval for a column's mean.
    
    Args:
        file_path: Path to the CSV file.
        column: Column to analyze.
        confidence_level: Confidence level (0.90, 0.95, or 0.99).
        
    Returns:
        Dict with confidence interval bounds and interpretation.
    """
    try:
        from scipy import stats
        
        df = load_csv(file_path)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        data = safe_numeric_series(df[column]).dropna()
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for confidence interval")
        
        n = len(data)
        mean = data.mean()
        std_err = data.std() / np.sqrt(n)
        
        # Calculate critical value for t-distribution
        alpha = 1 - confidence_level
        df_val = n - 1
        t_crit = stats.t.ppf(1 - alpha/2, df_val)
        
        margin_of_error = t_crit * std_err
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        result = {
            "column": column,
            "sample_size": n,
            "mean": round(float(mean), 4),
            "std_dev": round(float(data.std()), 4),
            "std_error": round(float(std_err), 6),
            "confidence_level": confidence_level,
            "confidence_interval": {
                "lower_bound": round(float(ci_lower), 4),
                "upper_bound": round(float(ci_upper), 4),
                "margin_of_error": round(float(margin_of_error), 4),
            },
            "interpretation": (
                f"We are {int(confidence_level*100)}% confident that the true population mean "
                f"lies between {ci_lower:.2f} and {ci_upper:.2f}."
            ),
        }
        
        return result
    
    except ImportError:
        return {"error": "scipy not installed; install with: pip install scipy"}
    except Exception as e:
        return {"error": str(e), "column": column}


# ==============================================================================
# BUSINESS METRICS & KPI CALCULATION
# ==============================================================================

def calculate_business_kpis(
    file_path: str,
    date_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate key business KPIs (customer metrics, conversion, efficiency ratios).
    
    Args:
        file_path: Path to the CSV file.
        date_column: Column name for dates; auto-detected if not provided.
        
    Returns:
        Dict with calculated KPIs and business health metrics.
    """
    try:
        df = load_csv(file_path)
        
        kpis = {
            "calculated_kpis": [],
            "summary": {},
        }
        
        # Revenue metrics
        if "revenue" in df.columns:
            revenue = safe_numeric_series(df["revenue"])
            total_revenue = revenue.sum()
            avg_transaction = revenue.mean()
            max_transaction = revenue.max()
            revenue_per_row = total_revenue / len(df) if len(df) > 0 else 0
            
            kpis["calculated_kpis"].append({
                "name": "Total Revenue",
                "value": round(total_revenue, 2),
                "category": "Revenue",
            })
            kpis["calculated_kpis"].append({
                "name": "Average Transaction Value",
                "value": round(avg_transaction, 2),
                "category": "Revenue",
            })
            kpis["calculated_kpis"].append({
                "name": "Revenue per Record",
                "value": round(revenue_per_row, 2),
                "category": "Revenue",
            })
        
        # Quantity/Volume metrics
        if "quantity" in df.columns:
            quantity = safe_numeric_series(df["quantity"])
            total_quantity = quantity.sum()
            avg_quantity = quantity.mean()
            
            kpis["calculated_kpis"].append({
                "name": "Total Units Sold",
                "value": int(total_quantity),
                "category": "Volume",
            })
            kpis["calculated_kpis"].append({
                "name": "Average Units per Transaction",
                "value": round(avg_quantity, 2),
                "category": "Volume",
            })
        
        # Category performance
        if "category" in df.columns and "revenue" in df.columns:
            revenue_by_category = df.groupby("category")["revenue"].sum()
            total_revenue = df["revenue"].sum()
            top_category = revenue_by_category.idxmax()
            top_category_share = (revenue_by_category.max() / total_revenue * 100) if total_revenue > 0 else 0
            
            kpis["calculated_kpis"].append({
                "name": "Top Category Revenue Share",
                "value": round(top_category_share, 2),
                "unit": "%",
                "category": "Performance",
                "detail": f"Category: {top_category}",
            })
            
            kpis["calculated_kpis"].append({
                "name": "Category Diversity",
                "value": len(revenue_by_category),
                "category": "Performance",
                "detail": f"Number of distinct categories",
            })
        
        # Regional/Segment performance
        if "region" in df.columns and "revenue" in df.columns:
            revenue_by_region = df.groupby("region")["revenue"].sum()
            total_revenue = df["revenue"].sum()
            top_region = revenue_by_region.idxmax()
            top_region_share = (revenue_by_region.max() / total_revenue * 100) if total_revenue > 0 else 0
            
            kpis["calculated_kpis"].append({
                "name": "Top Region Revenue Share",
                "value": round(top_region_share, 2),
                "unit": "%",
                "category": "Performance",
                "detail": f"Region: {top_region}",
            })
        
        # Time-based metrics
        if date_column is None:
            date_column = detect_date_column(df)
        
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            valid_dates = df[date_column].dropna()
            
            if len(valid_dates) > 0 and "revenue" in df.columns:
                date_range_days = (valid_dates.max() - valid_dates.min()).days + 1
                if date_range_days > 0:
                    daily_avg_revenue = df["revenue"].sum() / date_range_days
                    kpis["calculated_kpis"].append({
                        "name": "Daily Average Revenue",
                        "value": round(daily_avg_revenue, 2),
                        "category": "Time-based",
                    })
        
        # Product metrics
        if "product" in df.columns:
            unique_products = df["product"].nunique()
            kpis["calculated_kpis"].append({
                "name": "Number of Products",
                "value": unique_products,
                "category": "Portfolio",
            })
        
        # Data quality KPI
        completeness = (1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))) * 100 if len(df) > 0 else 0
        kpis["calculated_kpis"].append({
            "name": "Data Completeness",
            "value": round(completeness, 2),
            "unit": "%",
            "category": "Quality",
        })
        
        kpis["summary"] = {
            "total_records": len(df),
            "kpis_calculated": len(kpis["calculated_kpis"]),
            "categories": list(set(kpi.get("category", "Other") for kpi in kpis["calculated_kpis"])),
        }
        
        return kpis
    
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# REPORT EXPORT CAPABILITIES
# ==============================================================================

def export_report_to_excel(
    report_dict: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    """Export a comprehensive report dictionary to Excel format.
    
    Args:
        report_dict: The analysis report dictionary (typically from generate_comprehensive_report).
        output_path: Path where the Excel file should be saved.
        
    Returns:
        Dict with export status and file location.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        
        output_path = resolve_path(output_path)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            if "executive_summary" in report_dict and isinstance(report_dict["executive_summary"], list):
                summary_df = pd.DataFrame({
                    "Executive Summary": report_dict["executive_summary"]
                })
                summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)
            
            # Sheet 2: Data Quality
            if "quality_report" in report_dict and isinstance(report_dict["quality_report"], dict):
                quality_data = []
                for key, value in report_dict["quality_report"].items():
                    if not isinstance(value, (list, dict)):
                        quality_data.append({"Metric": key, "Value": value})
                if quality_data:
                    quality_df = pd.DataFrame(quality_data)
                    quality_df.to_excel(writer, sheet_name="Data Quality", index=False)
            
            # Sheet 3: Segment Comparisons
            if "segment_comparisons" in report_dict:
                seg_dict = report_dict["segment_comparisons"]
                if isinstance(seg_dict, dict) and "segment_results" in seg_dict:
                    seg_df = pd.DataFrame(list(seg_dict["segment_results"].items()), columns=["Segment", "Value"])
                    seg_df.to_excel(writer, sheet_name="Segment Analysis", index=False)
            
            # Sheet 4: Anomalies
            if "anomaly_detection" in report_dict:
                anom_dict = report_dict["anomaly_detection"]
                if isinstance(anom_dict, dict):
                    anom_data = [{k: v for k, v in anom_dict.items() if not isinstance(v, (list, dict))}]
                    anom_df = pd.DataFrame(anom_data).T
                    anom_df.to_excel(writer, sheet_name="Anomaly Detection")
            
            # Sheet 5: Key Drivers
            if "key_drivers" in report_dict:
                drivers_dict = report_dict["key_drivers"]
                if isinstance(drivers_dict, dict) and "top_drivers" in drivers_dict:
                    drivers_df = pd.DataFrame(drivers_dict["top_drivers"])
                    drivers_df.to_excel(writer, sheet_name="Key Drivers", index=False)
            
            # Sheet 6: Insights
            if "insights" in report_dict:
                insights_dict = report_dict["insights"]
                if isinstance(insights_dict, dict) and "findings" in insights_dict:
                    insights_df = pd.DataFrame({
                        "Insights": insights_dict["findings"]
                    })
                    insights_df.to_excel(writer, sheet_name="Insights", index=False)
        
        return {
            "status": "success",
            "file_path": output_path,
            "message": f"Report exported to Excel: {output_path}",
        }
    
    except ImportError:
        return {"error": "openpyxl not installed; install with: pip install openpyxl"}
    except Exception as e:
        return {"error": str(e)}


def export_report_to_pdf(
    report_dict: Dict[str, Any],
    output_path: str,
    title: str = "Data Analysis Report",
) -> Dict[str, Any]:
    """Export a comprehensive report dictionary to PDF format.
    
    Args:
        report_dict: The analysis report dictionary.
        output_path: Path where the PDF file should be saved.
        title: Title for the report.
        
    Returns:
        Dict with export status and file location.
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        output_path = resolve_path(output_path)
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        if "executive_summary" in report_dict:
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            if isinstance(report_dict["executive_summary"], list):
                for item in report_dict["executive_summary"]:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Data Quality
        if "quality_grade" in report_dict:
            story.append(Paragraph(f"Data Quality Grade: <b>{report_dict['quality_grade']}</b>", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Key Metrics
        if "key_drivers" in report_dict and isinstance(report_dict["key_drivers"], dict):
            drivers = report_dict["key_drivers"]
            if "top_drivers" in drivers:
                story.append(Paragraph("Top Drivers", styles['Heading2']))
                for driver in drivers["top_drivers"]:
                    story.append(Paragraph(
                        f"• {driver.get('feature', 'N/A')}: {driver.get('correlation', 'N/A')}",
                        styles['Normal']
                    ))
                story.append(Spacer(1, 0.2*inch))
        
        # Insights
        if "insights" in report_dict and isinstance(report_dict["insights"], dict):
            insights = report_dict["insights"]
            if "findings" in insights:
                story.append(Paragraph("Key Findings", styles['Heading2']))
                for finding in insights["findings"][:5]:  # Top 5 findings
                    story.append(Paragraph(f"• {finding}", styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        
        return {
            "status": "success",
            "file_path": output_path,
            "message": f"Report exported to PDF: {output_path}",
        }
    
    except ImportError:
        return {"error": "reportlab not installed; install with: pip install reportlab"}
    except Exception as e:
        return {"error": str(e)}