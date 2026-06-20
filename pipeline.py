from crewai import Task, Crew, Process
from agents.planner_agent import planner_agent
from agents.data_prep_agent import data_prep_agent
from agents.modeling_agent import modeling_agent
from agents.evaluator_agent import evaluator_agent


def _safe_output(task) -> str:
    try:
        return task.output.raw if task.output else ""
    except Exception:
        return str(task.output) if task.output else ""


def run_pipeline(file_path: str, target_column: str = "", document_path: str = "") -> dict:
    planner = planner_agent()
    data_prep = data_prep_agent()
    modeling = modeling_agent()
    evaluator = evaluator_agent()

    target_hint = (
        f"The target column is '{target_column}'."
        if target_column
        else "No target column specified — if data has no obvious label, choose clustering."
    )

    document_hint = (
        f"A dataset description document is available at '{document_path}'. FIRST call the "
        "Dataset Document Reader tool on it to learn dataset-specific conventions (such as how "
        "missing values are encoded, e.g. -200 or -999). Use what you learn to inform your analysis."
        if document_path
        else "No dataset description document was provided; rely on the data tools alone."
    )

    planner_task = Task(
        description=(
            f"Use the CSV Inspector and Data Insight Analyzer tools on '{file_path}' to thoroughly "
            f"understand the dataset before deciding the problem type. {document_hint} {target_hint} "
            "Consider: unique value ratios per column, presence of outliers, high correlations, "
            "and whether a clear target column exists. If the description document reveals special "
            "missing-value codes, explicitly note them in your report. "
            "Justify your decision with specific statistics."
        ),
        expected_output=(
            "A structured report with: (1) dataset summary stats, (2) key insights from correlation/outlier analysis, "
            "(3) any dataset-specific conventions found in the description document (if provided), "
            "(4) final problem type decision (classification/regression/clustering) with evidence-based justification."
        ),
        agent=planner,
    )

    doc_arg = (
        f"When calling the Data Cleaner tool, pass document_path='{document_path}' so it can read "
        "the dataset description and detect documented missing-value codes. "
        if document_path
        else "Call the Data Cleaner tool; if no description document exists it will use AI reasoning "
             "to detect missing-value sentinels from column names and statistics. "
    )

    data_prep_task = Task(
        description=(
            f"Clean the dataset at '{file_path}'. {doc_arg}"
            "The cleaner detects and neutralizes missing-value sentinels (e.g. -200), handles "
            "missing values, and encodes categorical columns, then saves the cleaned data. "
            "Report exactly what the cleaner did, including any sentinel handling."
        ),
        expected_output=(
            "A confirmation that the data was cleaned and saved, including the final shape, "
            "column list, and a note of any missing-value sentinels that were detected and imputed."
        ),
        agent=data_prep,
    )

    target_instruction = (
        f"The target column is '{target_column}'."
        if target_column
        else "This is a clustering problem — do not specify a target column when calling the tool."
    )

    modeling_task = Task(
        description=(
            "Train a machine learning model on the cleaned dataset at 'data/cleaned.csv'. "
            f"Use the problem type identified by the planner. {target_instruction} "
            "Evaluate the model and produce a feature importance or cluster visualization."
        ),
        expected_output="A summary of the trained model, its evaluation score, and the path to the visualization.",
        agent=modeling,
        context=[planner_task],
    )

    evaluator_task = Task(
        description=(
            "FIRST call the Model Evaluation Reporter tool to retrieve the actual model metrics and "
            "dataset statistics. Then, using those numbers alongside context from all previous agents, "
            "produce a structured evaluation report covering: "
            "(1) Problem type correctness, "
            "(2) Data cleaning quality (missing values, encoding), "
            "(3) Model performance verdict (is the score acceptable for this domain?), "
            "(4) Top 3 ranked improvement suggestions with expected impact. "
            "End with a clear PASS / NEEDS IMPROVEMENT / FAIL verdict."
        ),
        expected_output=(
            "A structured evaluation report with quantitative evidence from the evaluation tool, "
            "assessment of each pipeline stage, and a final verdict with ranked improvement suggestions."
        ),
        agent=evaluator,
        context=[planner_task, data_prep_task, modeling_task],
    )

    crew = Crew(
        agents=[planner, data_prep, modeling, evaluator],
        tasks=[planner_task, data_prep_task, modeling_task, evaluator_task],
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()

    return {
        "planner": _safe_output(planner_task),
        "data_prep": _safe_output(data_prep_task),
        "modeling": _safe_output(modeling_task),
        "evaluator": _safe_output(evaluator_task),
    }