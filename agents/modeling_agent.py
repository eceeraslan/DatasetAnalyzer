from crewai import Agent
from config.llm import llm
from tools.modeling_tool import ModelingTool


def modeling_agent():
    return Agent(
        role="Modeling and Insight Analyst",
        goal="Train an appropriate machine learning model based on the problem type, evaluate its performance, and produce a feature importance visualization.",
        backstory="You are a skilled machine learning practitioner who selects the right model for the task, measures its performance honestly, and surfaces which features drive the predictions.",
        llm=llm,
        verbose=True,
        tools=[ModelingTool()]
    )