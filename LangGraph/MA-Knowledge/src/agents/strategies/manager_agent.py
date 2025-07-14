# src/agents/manager_agent.py
from typing import Dict, Any, Union
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from src.agents.base_agent import Analyst

class ManagerAgent(Analyst):
    """Manager analyst with access to all tools including LLM-based knowledge generation"""
    
    def __init__(self):
        super().__init__(
            name="Manager",
            description="A manager analyst with access to all tools including LLM-based knowledge generation.",
            tools=["arxiv", "wikipedia"],
            system_prompt = """
                You are **Manager Planner**, an advanced agent that decides which actions to take for complex image-based Q&A tasks requiring reasoning and external knowledge.

                **Available Actions**  
                - **Action_1:** Perform Visual Question Answering (VQA) on the image.  
                - **Action_2:** Retrieve background knowledge from arXiv or Wikipedia.  
                - **Action_3:** Generate contextual insights using a Large Language Model (LLM).

                **Rules**  
                1. **Always** begin with **Action_1**. 
                2. Add **Action_2** if the question requires factual or external knowledge.  
                3. Add **Action_3** if deeper explanation, background, or reasoning is likely needed.

                **Input**  
                - **Context:** `{context}`  
                - **Question:** `{question}`  

                **Output**  
                Response format:  [Action_1, Action_2, Action_3]  
            """,
            final_system_prompt="""
                Please answer the question according to the context and candidate answers. 
                ====== 
                Context: A close up of an elephant standing behind a cement wall. 
                Question: What item in the picture is purported to have a great memory? 
                Candidates: elephant(0.99), trunk(0.70), dumbo(0.09), brain(0.08), tusk(0.03) 
                Answer: elephant 
                ====== 
                Context: {context}. 
                Question: {question}. 
                Candidates: {candidates}. 
                Answer: 
        """
        )

def create_manager_agent() -> ManagerAgent:
    """Factory function to create manager agent"""
    return ManagerAgent()