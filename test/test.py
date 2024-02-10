from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

llm = Ollama(model="mistral")

python_tool = Tool(
    name="python_tool",
    description="Python shell that executed Python commands.",
    func=PythonREPL().run
)

agent = Agent(
    role='Data Cleaner',
    goal='Import and clean data',
    backstory='An experienced data analyst that is highly skilled in Python.',
    llm = llm,
    vebose=True,
    tools=[python_tool]
)

task = Task(
    agent=agent,
    description='Print out a print hello statement in Python.'
)

task.execute()