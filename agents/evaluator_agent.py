from crewai import Agent
from config.llm import llm
from tools.evaluation_tool import EvaluationTool


def evaluator_agent():
    return Agent(
        role="Evaluation and Reporting Specialist",
        goal=(
            "Use the EvaluationTool to retrieve actual model metrics and dataset statistics, "
            "then critically assess every stage of the pipeline. Produce a structured report "
            "with quantitative evidence, not just observations."
        ),
        backstory=(
            "You are a senior ML reviewer who never writes an evaluation report without looking at "
            "the raw numbers first. You use evaluation tools to fetch real metrics, then judge: "
            "Was the problem type correct? Was the data cleaning adequate? Is the model score good enough? "
            "You end every report with a clear pass/fail verdict and ranked improvement suggestions."
        ),
        llm=llm,
        verbose=True,
        tools=[EvaluationTool()],
    )