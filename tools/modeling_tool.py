import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from crewai.tools import BaseTool
from tools.io_utils import smart_read_csv


def _save_correlation_heatmap(frame, path):
    """Saves a correlation heatmap of the numeric columns to `path`."""
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


class ModelingTool(BaseTool):
    name: str = "Modeling and Insight Tool"
    description: str = (
        "Trains a machine learning model on the cleaned dataset based on the problem type "
        "and evaluates it. For classification/regression, requires target_column. "
        "For clustering, leave target_column empty."
    )

    def _run(self, problem_type: str, target_column: str = "") -> str:
        os.makedirs("data", exist_ok=True)
        df = smart_read_csv("data/cleaned.csv")
        pt = problem_type.lower().strip()
        tc = target_column.strip()
        plot_path = "data/feature_importance.png"

        # --- CLUSTERING ---
        if pt == "clustering" or not tc or tc.lower() in ("none", "n/a", "na", ""):
            # If a target column was provided, exclude it from clustering so the
            # label does not leak into the features being grouped.
            col_map = {c.lower(): c for c in df.columns}
            cluster_df = df
            if tc and tc.lower() in col_map:
                cluster_df = df.drop(columns=[col_map[tc.lower()]])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cluster_df)

            best_k, best_score = 2, -1
            for k in range(2, min(8, len(cluster_df))):
                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score, best_k = score, k

            final_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

            plt.figure(figsize=(8, 5))
            pd.Series(final_labels).value_counts().sort_index().plot(kind="bar", color="steelblue")
            plt.xlabel("Cluster")
            plt.ylabel("Count")
            plt.title(f"KMeans Clustering — k={best_k}")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            heatmap_path = _save_correlation_heatmap(cluster_df, "data/correlation_heatmap.png")

            results = {
                "problem_type": "clustering",
                "optimal_k": best_k,
                "silhouette_score": round(best_score, 4),
                "plot_path": plot_path,
                "heatmap_path": heatmap_path,
            }
            with open("data/model_results.json", "w") as f:
                json.dump(results, f, indent=2)

            return (
                f"Clustering complete.\n"
                f"Optimal clusters (k): {best_k}\n"
                f"Silhouette Score: {best_score:.4f}\n"
                f"Cluster chart saved to: {plot_path}\n"
                f"Correlation heatmap saved to: {heatmap_path}"
            )

        # --- CLASSIFICATION / REGRESSION ---
        col_map = {c.lower(): c for c in df.columns}
        actual_col = col_map.get(tc.lower())
        if actual_col is None:
            return f"Target column '{tc}' not found. Available columns: {list(df.columns)}"

        X = df.drop(columns=[actual_col])
        y = df[actual_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Candidate models per problem type — the agent compares them and
        # selects the best by cross-validated score (more honest than a single split).
        if pt == "classification":
            candidates = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }
            scoring, metric_name = "accuracy", "Accuracy"
            final_metric = accuracy_score
        elif pt == "regression":
            candidates = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            }
            scoring, metric_name = "r2", "R² Score"
            final_metric = r2_score
        else:
            return f"Unknown problem type: '{problem_type}'. Use 'classification', 'regression', or 'clustering'."

        # Cross-validate each candidate and keep the leaderboard
        n_splits = max(2, min(5, int(y.value_counts().min()) if pt == "classification" else 5))
        leaderboard = {}
        for cname, cmodel in candidates.items():
            cv_scores = cross_val_score(cmodel, X, y, cv=n_splits, scoring=scoring)
            leaderboard[cname] = round(float(cv_scores.mean()), 4)

        best_name = max(leaderboard, key=leaderboard.get)
        model = candidates[best_name]
        model.fit(X_train, y_train)
        score = final_metric(y_test, model.predict(X_test))

        # Feature importance: tree models expose feature_importances_, linear
        # models expose coef_; fall back gracefully.
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(np.ravel(model.coef_))
        else:
            importances = np.zeros(len(X.columns))

        plt.figure(figsize=(10, 6))
        plt.barh(X.columns, importances)
        plt.xlabel("Importance")
        plt.title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # Correlation heatmap of the full feature set (+ target)
        heatmap_path = _save_correlation_heatmap(df, "data/correlation_heatmap.png")

        # Confusion matrix (classification only)
        confusion_path = None
        if pt == "classification":
            cm = confusion_matrix(y_test, model.predict(X_test))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            confusion_path = "data/confusion_matrix.png"
            plt.savefig(confusion_path)
            plt.close()

        results = {
            "problem_type": pt,
            "target_column": actual_col,
            "model": best_name,
            "model_leaderboard_cv": leaderboard,
            "cv_folds": n_splits,
            metric_name: round(float(score), 4),
            "n_features": len(X.columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "top_features": sorted(
                zip(X.columns, importances),
                key=lambda x: x[1], reverse=True
            )[:5],
            "plot_path": plot_path,
            "heatmap_path": heatmap_path,
            "confusion_path": confusion_path,
        }
        with open("data/model_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        leaderboard_str = ", ".join(f"{k}: {v}" for k, v in
                                    sorted(leaderboard.items(), key=lambda x: x[1], reverse=True))
        return (
            f"Modeling complete.\n"
            f"Problem type: {pt}\n"
            f"Target column: {actual_col}\n"
            f"Models compared ({n_splits}-fold CV): {leaderboard_str}\n"
            f"Selected best model: {best_name}\n"
            f"Test {metric_name}: {score:.4f}\n"
            f"Top features: {[f for f, _ in results['top_features']]}\n"
            f"Feature importance plot saved to: {plot_path}\n"
            f"Correlation heatmap saved to: {heatmap_path}\n"
            + (f"Confusion matrix saved to: {confusion_path}" if confusion_path else "")
        )