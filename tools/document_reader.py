"""Tool for extracting text from an optional dataset-description document.

Users may upload a PDF/TXT (or paste text) describing their dataset — e.g. a
UCI repository page stating that '-200' marks missing values. The Planner Agent
reads this context to make better-informed decisions instead of guessing.
"""
import os
from crewai.tools import BaseTool


def extract_document_text(path: str, max_chars: int = 8000) -> str:
    """Extracts plain text from a .pdf or .txt file. Returns '' if unreadable."""
    if not path or not os.path.exists(path):
        return ""

    ext = os.path.splitext(path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:  # treat everything else as plain text
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        return f"(Could not read document: {e})"

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


class DocumentReaderTool(BaseTool):
    name: str = "Dataset Document Reader"
    description: str = (
        "Reads an optional dataset-description document (PDF or TXT) and returns its text. "
        "Use this to understand dataset-specific conventions such as how missing values are "
        "encoded (e.g. -200, -999), units, or column meanings, before deciding on analysis steps."
    )

    def _run(self, document_path: str = "data/description.txt") -> str:
        text = extract_document_text(document_path)
        if not text:
            return "No dataset description document was provided."
        return f"Dataset description document contents:\n\n{text}"