from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(
    model="dashscope/qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)