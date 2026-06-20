import pandas as pd
from tools.io_utils import smart_read_csv
from crewai.tools import BaseTool

class CSVInspectorTool(BaseTool):
    name : str = "CSV Inspector"
    description: str = "Reads a CSV file and returns column names, data types, shape, and first few rows."

    def _run(self , file_path: str) -> str :
        df = smart_read_csv(file_path)
        info = f"Shape: {df.shape}\n"
        info +=  f"Columns: {list(df.columns)}\n"
        info +=  f"Data types:\n{df.dtypes}\n"
        info += f"First 3 rows:\n{df.head(3)}"
        return info