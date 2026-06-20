import pandas as pd
import numpy as np
from tools.io_utils import smart_read_csv
from crewai.tools import BaseTool


class InsightTool(BaseTool):
    name: str = "Data Insight Analyzer"
    description: str = (
        "Analyzes a CSV file to detect correlations, outliers, class distribution, and "
        "feature uniqueness. Use this to determine the best problem type and modeling strategy."
    )

    def _run(self, file_path: str) -> str:
        df = smart_read_csv(file_path)
        report = []

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

        report.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        report.append(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
        report.append(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

        # Missing value ratio
        missing = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing[missing > 10]
        if not high_missing.empty:
            report.append(f"High missing (>10%): {high_missing.to_dict()}")

        # Outlier detection via IQR
        outlier_summary = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            n = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            if n > 0:
                outlier_summary.append(f"{col}: {n}")
        if outlier_summary:
            report.append(f"Outliers detected: {', '.join(outlier_summary)}")

        # Top correlated pairs
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = upper.stack()
            high_corr = high_corr[high_corr > 0.7]
            if not high_corr.empty:
                pairs = [f"{a}↔{b}:{v:.2f}" for (a, b), v in high_corr.nlargest(3).items()]
                report.append(f"High correlations (>0.7): {', '.join(pairs)}")

        # Unique value counts — helps distinguish classification vs regression targets
        report.append("Unique values per column (potential targets):")
        for col in df.columns[:12]:
            n_unique = df[col].nunique()
            pct = n_unique / len(df)
            hint = "continuous→regression" if pct > 0.05 and pd.api.types.is_numeric_dtype(df[col]) and n_unique > 20 else \
                   "categorical→classification" if n_unique <= 20 else "high-cardinality"
            report.append(f"  {col}: {n_unique} unique ({pct:.1%}) — {hint}")

        return "\n".join(report)