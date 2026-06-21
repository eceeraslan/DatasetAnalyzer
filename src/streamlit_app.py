# Entry point for the application is app.py at the project root.
# Run: streamlit run app.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")) as _f:
    exec(_f.read())