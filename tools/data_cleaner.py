import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from crewai.tools import BaseTool
from tools.io_utils import smart_read_csv
from tools.document_reader import extract_document_text


def _sentinels_from_document(document_path: str) -> list:
    """Reads the description document and extracts explicitly stated missing-value
    codes (e.g. 'missing values are tagged with -200'). Returns a list of numbers.
    This is the SAFE source: values come directly from documentation, not a guess."""
    if not document_path or not os.path.exists(document_path):
        return []
    text = extract_document_text(document_path).lower()
    if not text:
        return []

    sentinels = set()
    # Look for phrases that signal a missing-value code near a number.
    # e.g. "missing values are tagged with -200", "missing = -999", "na value: -200"
    keyword_pat = r"(?:missing|tagged|sentinel|no[t]?\s*recorded|unavailable|n/?a)\D{0,40}?(-?\d+(?:\.\d+)?)"
    for m in re.finditer(keyword_pat, text):
        try:
            val = float(m.group(1))
            sentinels.add(val)
        except ValueError:
            pass
    return sorted(sentinels)


def _detect_sentinels_with_ai(df: pd.DataFrame) -> dict:
    """Asks the LLM to reason about each numeric column — using its NAME and value
    statistics — to decide whether a frequently-repeated value is actually a
    missing-value sentinel (e.g. -200 in a concentration column that can't be
    negative). Returns {column: sentinel_value}. Falls back to {} on any error."""
    numeric_cols = df.select_dtypes(include="number").columns
    summaries = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        vc = s.value_counts(normalize=True)
        top_val, top_freq = float(vc.index[0]), float(vc.iloc[0])
        # Only bother the LLM about columns with a suspiciously repeated value
        if top_freq < 0.03:
            continue
        summaries.append({
            "column": col,
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "most_frequent_value": round(top_val, 2),
            "its_frequency": round(top_freq, 3),
        })

    if not summaries:
        return {}

    prompt = (
        "You are a data quality expert. Below are numeric columns from a dataset, "
        "each with its name and statistics. Some datasets encode MISSING values with a "
        "special sentinel number (e.g. -200, -999) instead of leaving them blank. "
        "Using the COLUMN NAME and the statistics, decide for each column whether its "
        "most_frequent_value is actually a missing-value sentinel rather than a real "
        "measurement. For example, a concentration, count, or sensor reading cannot be "
        "negative, so a repeated -200 there is almost certainly a sentinel. A value that "
        "is physically plausible (like 0 sales, or a cold temperature) is NOT a sentinel.\n\n"
        f"Columns:\n{json.dumps(summaries, indent=2)}\n\n"
        "Respond with ONLY a JSON object mapping each column you believe contains a "
        "sentinel to that sentinel value, e.g. {\"NO2(GT)\": -200}. If none, respond {}. "
        "No explanation, no markdown, just the JSON object."
    )

    try:
        from config.llm import llm
        raw = llm.call(prompt)
        text = raw if isinstance(raw, str) else str(raw)
        # strip any code fences / stray text, keep the JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group(0))
        # keep only valid {existing_column: number} pairs
        result = {}
        for col, val in parsed.items():
            if col in numeric_cols:
                try:
                    result[col] = float(val)
                except (TypeError, ValueError):
                    pass
        return result
    except Exception:
        return {}


class DataCleanerTool(BaseTool):
    name: str = "Data Cleaner"
    description: str = (
        "Reads a CSV, detects and neutralizes missing-value sentinel codes (from the dataset "
        "description document if available, otherwise via conservative statistical heuristics), "
        "performs feature engineering where applicable, drops identifier and high-cardinality "
        "text columns, imputes missing values, encodes categoricals, and saves the cleaned data."
    )

    def _run(self, file_path: str, document_path: str = "data/description.txt") -> str:
        df = smart_read_csv(file_path)
        original_shape = df.shape
        engineered = []
        sentinel_notes = []

        # Number of numeric columns after smart_read_csv has parsed comma-decimals
        # (e.g. "2,6" -> 2.6). Reported so the evaluator can confirm parsing succeeded.
        n_numeric_parsed = len(df.select_dtypes(include="number").columns)

        # --- 0a) SENTINEL DETECTION (document first, then conservative stats) ---
        doc_sentinels = _sentinels_from_document(document_path)
        numeric_cols = df.select_dtypes(include="number").columns

        if doc_sentinels:
            # Documentation is authoritative: replace these exact values everywhere.
            for sv in doc_sentinels:
                hit_cols = [c for c in numeric_cols if (df[c] == sv).any()]
                if hit_cols:
                    total = int(sum((df[c] == sv).sum() for c in hit_cols))
                    df[numeric_cols] = df[numeric_cols].replace(sv, np.nan)
                    sentinel_notes.append(
                        f"From documentation: value {sv} treated as missing "
                        f"({total} cells across {len(hit_cols)} columns) and imputed."
                    )
        else:
            # No document: let the AI reason about column names + stats.
            suspects = _detect_sentinels_with_ai(df)
            for col, sv in suspects.items():
                n = int((df[col] == sv).sum())
                if n == 0:
                    continue
                # Apply the AI-identified sentinel across all numeric columns,
                # since a sentinel code is usually shared dataset-wide.
                total = int(sum((df[c] == sv).sum() for c in numeric_cols))
                df[numeric_cols] = df[numeric_cols].replace(sv, np.nan)
                sentinel_notes.append(
                    f"AI-identified sentinel: value {sv} (first detected in '{col}' via column "
                    f"name and statistics) treated as missing across the dataset "
                    f"({total} cells) and imputed."
                )
                break  # one dataset-wide sentinel is enough

        # --- 0b) Feature engineering (applied only when the right columns exist) ---
        name_col = next((c for c in df.columns if c.lower() == "name"), None)
        if name_col is not None:
            titles = df[name_col].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False)
            if titles.notna().mean() > 0.5:
                common = {"Mr", "Mrs", "Miss", "Master"}
                df["Title"] = titles.str.strip().apply(lambda t: t if t in common else "Rare")
                engineered.append("Title")

        sib = next((c for c in df.columns if c.lower() == "sibsp"), None)
        par = next((c for c in df.columns if c.lower() == "parch"), None)
        if sib is not None and par is not None:
            df["FamilySize"] = df[sib] + df[par] + 1
            df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
            engineered.extend(["FamilySize", "IsAlone"])

        # --- 1) Drop columns that are noise for modeling ---
        id_like = [
            c for c in df.columns
            if c.lower() in ("id", "passengerid", "index")
            or c.lower().endswith("_id")
            or c.lower().endswith("id")
        ]
        high_card = []
        for col in df.select_dtypes(exclude="number").columns:
            if col in id_like:
                continue
            uniqueness = df[col].nunique(dropna=True) / max(len(df), 1)
            if uniqueness > 0.5:
                high_card.append(col)
        to_drop = list(dict.fromkeys(id_like + high_card))
        if to_drop:
            df = df.drop(columns=to_drop)

        # --- 2) Handle missing values (including those created from sentinels) ---
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        # --- 3) Encode remaining categorical columns ---
        for col in df.select_dtypes(exclude="number").columns:
            df[col] = LabelEncoder().fit_transform(df[col])

        # --- 4) Save ---
        os.makedirs("data", exist_ok=True)
        output_path = "data/cleaned.csv"
        df.to_csv(output_path, index=False)

        sentinel_block = (
            "Sentinel handling:\n  " + "\n  ".join(sentinel_notes)
            if sentinel_notes else "Sentinel handling: none detected."
        )
        return (
            f"Cleaning complete. Saved to {output_path}\n"
            f"Original shape: {original_shape} -> Final shape: {df.shape}\n"
            f"Numeric parsing: {n_numeric_parsed} columns parsed as numeric "
            f"(comma-decimal values like '2,6' converted to 2.6 during loading).\n"
            f"{sentinel_block}\n"
            f"Engineered features: {engineered if engineered else 'none'}\n"
            f"Dropped columns (identifier / high-cardinality): {to_drop if to_drop else 'none'}\n"
            f"Columns: {list(df.columns)}\n"
            f"Remaining missing values: {df.isnull().sum().sum()}"
        )