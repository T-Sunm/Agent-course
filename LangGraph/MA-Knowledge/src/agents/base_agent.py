from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

class Analyst(BaseModel):
    """Base model for all analysts"""
    name: str = Field(description="Name of the analyst.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    tools: List[str]
    system_prompt: str = Field(description="System prompt for the analyst.")
    final_system_prompt: str = Field(default="", description="Final system prompt for reasoning.")
    
    @property
    def affiliation(self) -> str:
        return f"{self.name} Agent for VQA"
    
    @property
    def persona(self) -> str:
        return f"Name: {self.affiliation}\nTools: {self.tools}\nDescription: {self.description}"
    