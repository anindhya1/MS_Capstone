from agents import Agent
from pydantic import BaseModel, Field


INSTRUCTIONS = ""

class evaluator_agent(BaseModel):
    feedback: str = Field("layout for output from LLM(feedback)")


evaluator_agent = Agent(name= "", instructions= "", model= "", output_type="")
