import json
import os
import pandas as pd
from tools.io_utils import smart_read_csv
from crewai.tools import BaseTool


class EvaluationTool(BaseTool):
    name: str = "Model Evaluation Reporter"
    description: str = (
        "Reads the saved model results and the cleaned dataset (from their fixed paths) to "
        "produce a quantitative evaluation report. Takes no arguments. Always call this tool "
        "before writing your final evaluation."
    )

    def _run(self, file_path: str = "data/cleaned.csv") -> str:
        report = []

        # The cleaned dataset is always at this fixed path. Ignore any other path
        # the caller may pass (e.g. the results JSON), so stats are never read
        # from the wrong file.
        cleaned_path = "data/cleaned.csv"

        # Model metrics
        results_path = "data/model_results.json"
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            report.append("=== Model Results ===")
            for k, v in results.items():
                report.append(f"  {k}: {v}")
        else:
            report.append("WARNING: No model_results.json found — modeling may not have completed.")

        # Dataset quality
        if os.path.exists(cleaned_path):
            df = smart_read_csv(cleaned_path)
            report.append("\n=== Cleaned Dataset Stats ===")
            report.append(f"  Shape: {df.shape}")
            report.append(f"  Remaining missing values: {df.isnull().sum().sum()}")
            report.append(f"  Features: {list(df.columns)}")
            numeric = df.select_dtypes(include="number")
            report.append(f"  Numeric features: {len(numeric.columns)}")
            report.append("  Feature value ranges:")
            for col in numeric.columns[:6]:
                report.append(f"    {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
        else:
            report.append(f"WARNING: {cleaned_path} not found.")

        # Visualization check
        plot_path = "data/feature_importance.png"
        if os.path.exists(plot_path):
            size_kb = os.path.getsize(plot_path) // 1024
            report.append(f"\n=== Visualization ===\n  feature_importance.png present ({size_kb} KB)")
        else:
            report.append("\nWARNING: feature_importance.png not found.")

        return "\n".join(report)