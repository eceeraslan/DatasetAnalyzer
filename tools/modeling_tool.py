import json
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, r2_score, silhouette_score, confusion_matrix,
    f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from crewai.tools import BaseTool
from tools.io_utils import smart_read_csv
from tools.session import get_data_dir


def _save_correlation_heatmap(frame, path):
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


def _build_preprocessor(X):
    """ColumnTransformer: StandardScaler for numerics, OneHotEncoder for categoricals.
    Fitted inside Pipeline per CV fold — no data leakage."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    transformers = [("num", StandardScaler(), numeric_cols)]
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        )
    return ColumnTransformer(transformers=transformers)


def _feature_names(pipeline, X):
    return pipeline.named_steps["prep"].get_feature_names_out().tolist()


class ModelingTool(BaseTool):
    name: str = "Modeling and Insight Tool"
    description: str = (
        "Trains a machine learning model on the cleaned dataset based on the problem type "
        "and evaluates it. For classification/regression, requires target_column. "
        "For clustering, leave target_column empty."
    )

    def _run(self, problem_type: str, target_column: str = "") -> str:
        data_dir = get_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        df = smart_read_csv(os.path.join(data_dir, "cleaned.csv"))
        pt = problem_type.lower().strip()
        tc = target_column.strip()
        plot_path = os.path.join(data_dir, "feature_importance.png")

        # --- CLUSTERING ---
        if pt == "clustering" or not tc or tc.lower() in ("none", "n/a", "na", ""):
            col_map = {c.lower(): c for c in df.columns}
            cluster_df = df.drop(columns=[col_map[tc.lower()]]) if tc and tc.lower() in col_map else df

            preprocessor = _build_preprocessor(cluster_df)
            X_scaled = preprocessor.fit_transform(cluster_df)

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

            heatmap_path = os.path.join(data_dir, "correlation_heatmap.png")
            _save_correlation_heatmap(cluster_df, heatmap_path)

            results = {
                "problem_type": "clustering",
                "optimal_k": best_k,
                "silhouette_score": round(best_score, 4),
                "plot_path": plot_path,
                "heatmap_path": heatmap_path,
            }
            with open(os.path.join(data_dir, "model_results.json"), "w") as f:
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

        # Stratified split preserves class distribution for classification
        stratify = y if pt == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        if pt == "classification":
            candidates = {
                "Logistic Regression": Pipeline([
                    ("prep", _build_preprocessor(X)), ("model", LogisticRegression(max_iter=1000))
                ]),
                "Random Forest": Pipeline([
                    ("prep", _build_preprocessor(X)), ("model", RandomForestClassifier(random_state=42))
                ]),
                "Gradient Boosting": Pipeline([
                    ("prep", _build_preprocessor(X)), ("model", GradientBoostingClassifier(random_state=42))
                ]),
            }
            scoring, metric_name = "accuracy", "Accuracy"
            final_metric = accuracy_score
        elif pt == "regression":
            candidates = {
                "Linear Regression": Pipeline([
                    ("prep", _build_preprocessor(X)), ("model", LinearRegression())
                ]),
                "Random Forest": Pipeline([
                    ("prep", _build_preprocessor(X)), ("model", RandomForestRegressor(random_state=42))
                ]),
                "Gradient Boosting": Pipeline([
                    ("prep", _build_preprocessor(X)), ("model", GradientBoostingRegressor(random_state=42))
                ]),
            }
            scoring, metric_name = "r2", "R² Score"
            final_metric = r2_score
        else:
            return f"Unknown problem type: '{problem_type}'. Use 'classification', 'regression', or 'clustering'."

        # Pipeline-wrapped CV: preprocessor fit per fold, never leaks test data
        n_splits = max(2, min(5, int(y.value_counts().min()) if pt == "classification" else 5))
        leaderboard = {}
        for cname, pipe in candidates.items():
            cv_scores = cross_val_score(pipe, X, y, cv=n_splits, scoring=scoring)
            leaderboard[cname] = round(float(cv_scores.mean()), 4)

        best_name = max(leaderboard, key=leaderboard.get)
        model = candidates[best_name]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = final_metric(y_test, preds)

        # Extended metrics per problem type
        extra_metrics = {}
        if pt == "classification":
            extra_metrics["F1 (weighted)"] = round(float(f1_score(y_test, preds, average="weighted")), 4)
            extra_metrics["Precision (weighted)"] = round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4)
            extra_metrics["Recall (weighted)"] = round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4)
        elif pt == "regression":
            extra_metrics["RMSE"] = round(float(math.sqrt(mean_squared_error(y_test, preds))), 4)
            extra_metrics["MAE"] = round(float(mean_absolute_error(y_test, preds)), 4)

        # Persist fitted pipeline so users can run predictions on new data
        model_path = os.path.join(data_dir, "model.pkl")
        joblib.dump(model, model_path)

        # Feature importance (expanded OHE names, e.g. cat__Sex_female, num__Age)
        feat_names = _feature_names(model, X)
        inner = model.named_steps["model"]
        if hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            importances = np.abs(np.ravel(inner.coef_))
        else:
            importances = np.zeros(len(feat_names))

        feat_names = feat_names[:len(importances)]

        plt.figure(figsize=(10, 6))
        plt.barh(feat_names, importances)
        plt.xlabel("Importance")
        plt.title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        heatmap_path = os.path.join(data_dir, "correlation_heatmap.png")
        _save_correlation_heatmap(df, heatmap_path)

        confusion_path = None
        if pt == "classification":
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            confusion_path = os.path.join(data_dir, "confusion_matrix.png")
            plt.savefig(confusion_path)
            plt.close()

        top_features = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:5]

        results = {
            "problem_type": pt,
            "target_column": actual_col,
            "model": best_name,
            "model_leaderboard_cv": leaderboard,
            "cv_folds": n_splits,
            metric_name: round(float(score), 4),
            **extra_metrics,
            "n_features": len(feat_names),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "top_features": top_features,
            "plot_path": plot_path,
            "heatmap_path": heatmap_path,
            "confusion_path": confusion_path,
            "model_path": model_path,
        }
        with open(os.path.join(data_dir, "model_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        leaderboard_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
        )
        extra_str = "  |  ".join(f"{k}: {v}" for k, v in extra_metrics.items())
        return (
            f"Modeling complete.\n"
            f"Problem type: {pt}\n"
            f"Target column: {actual_col}\n"
            f"Models compared ({n_splits}-fold CV): {leaderboard_str}\n"
            f"Selected best model: {best_name}\n"
            f"Test {metric_name}: {score:.4f}  |  {extra_str}\n"
            f"Top features: {[f for f, _ in top_features]}\n"
            f"Feature importance plot saved to: {plot_path}\n"
            f"Correlation heatmap saved to: {heatmap_path}\n"
            f"Trained model saved to: {model_path}\n"
            + (f"Confusion matrix saved to: {confusion_path}" if confusion_path else "")
        )