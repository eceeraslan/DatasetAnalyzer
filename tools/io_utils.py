"""Shared CSV reading helper.

Real-world CSVs are not always comma-separated with dot decimals. This reader
auto-detects the delimiter, handles European-style (semicolon + comma-decimal)
files, and drops fully empty 'Unnamed' columns so the rest of the pipeline can
assume a clean, well-formed DataFrame.
"""
import pandas as pd


def smart_read_csv(file_path) -> pd.DataFrame:
    # 1) Let pandas sniff the delimiter (engine="python" + sep=None).
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path)

    # 2) If sniffing failed (everything landed in one column) and that single
    #    header contains ';', it's almost certainly a semicolon file with
    #    comma decimals — re-read explicitly.
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(file_path, sep=";", decimal=",")

    # 3) Drop fully empty columns (e.g. trailing 'Unnamed: 15' from a ';;' line).
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # 4) Drop fully empty rows.
    df = df.dropna(axis=0, how="all")

    # 5) Recover numeric columns that were left as text because of comma decimals
    #    (e.g. "2,6" -> 2.6). Only convert a column if doing so succeeds for the
    #    vast majority of its non-null values, so genuine text columns are untouched.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            non_null = df[col].notna().sum()
            if non_null > 0 and converted.notna().sum() >= 0.9 * non_null:
                df[col] = converted

    return df