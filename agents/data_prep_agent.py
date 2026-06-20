from crewai import Agent
from config.llm import llm
from tools.data_cleaner import DataCleanerTool


def data_prep_agent():
    return Agent(
        role="Data Preparation Specialist",
        goal="Clean the given dataset by handling missing values and encoding categorical columns, then save the cleaned version to disk.",
        backstory="You are a meticulous data engineer who prepares raw datasets for machine learning. You ensure data is complete, properly typed, and ready for modeling.",
        llm=llm,
        verbose=True,
        tools=[DataCleanerTool()]
    )