from crewai import Agent
from config.llm import llm
from tools.csv_inspector import CSVInspectorTool
from tools.insight_tool import InsightTool
from tools.document_reader import DocumentReaderTool


def planner_agent():
    return Agent(
        role="Data Science Planner",
        goal=(
            "Analyze the given CSV dataset using both structural inspection and statistical insight. "
            "If a dataset description document is available, read it first to learn dataset-specific "
            "conventions (such as how missing values are encoded, e.g. -200 or -999, units, or column "
            "meanings). Decide the correct problem type (classification, regression, or clustering) based on "
            "column types, unique value ratios, correlations, outlier patterns, and any documented conventions."
        ),
        backstory=(
            "You are a senior data scientist who makes well-informed decisions about ML problem framing. "
            "You use multiple tools to understand a dataset deeply before recommending an approach. "
            "When the user provides a dataset description, you always read it, because documentation often "
            "reveals things raw data cannot — like sentinel values that secretly mark missing data. "
            "You consider class imbalance, feature correlations, and data quality before deciding."
        ),
        llm=llm,
        verbose=True,
        tools=[CSVInspectorTool(), InsightTool(), DocumentReaderTool()],
    )