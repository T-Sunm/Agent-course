from typing import Annotated, List, Dict, Any
from langgraph.graph import MessagesState
from src.agents.base_agent import Analyst
from src.agents.strategies.junior_agent import JuniorAgent
from src.agents.strategies.senior_agent import SeniorAgent
from src.agents.strategies.manager_agent import ManagerAgent
import operator

class ViReAgentState(MessagesState):
    question: str
    image: str
    image_caption: str
    results: Annotated[List[str], operator.add]
    final_answer: str
    voting_details: Dict[str, Any]


class ViReJuniorState(MessagesState):
    question: str
    image: str
    analyst: JuniorAgent
    number_of_steps: int
    answer_candidate: str

class ViReSeniorState(MessagesState):
    question: str
    image: str
    analyst: SeniorAgent
    number_of_steps: int
    answer_candidate: str
    KBs_Knowledge: Annotated[List[str], operator.add]

class ViReManagerState(MessagesState):
    question: str
    image: str
    analyst: ManagerAgent
    number_of_steps: int
    answer_candidate: str
    KBs_Knowledge: Annotated[List[str], operator.add]
    LLM_Knowledge: str
