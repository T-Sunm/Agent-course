from typing import Union, Dict, Any
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from src.core.state import ViReJuniorState, ViReSeniorState, ViReManagerState
from src.models.llm_provider import get_llm
from src.utils.tools_utils import _process_knowledge_result
import re

def tool_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState], 
              tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Process tool calls and update state"""
    outputs = []
    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    print(f"Processing {len(tool_calls)} tool calls")
    
    updates = {"messages": outputs}

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        try:
            if tool_name == "vqa_tool":
                print(f"Processing vqa_tool calls")
                args = tool_call["args"]
                if not args.get("image_url"):
                    args["image_url"] = state.get("image")
                result = tools_registry[tool_name].invoke(args)
                print(f"Result: {result}")
                updates["answer_candidate"] = result
                
            elif tool_name in ["arxiv", "wikipedia"]:
                print(f"Processing {tool_name} calls")
                raw_result = tools_registry[tool_name].invoke(tool_call["args"])
                print(f"Raw Result: {raw_result}")
                
                # Process and format the result
                processed_result = _process_knowledge_result(raw_result, tool_name)
                
                if "KBs_Knowledge" not in updates:
                    updates["KBs_Knowledge"] = []
                updates["KBs_Knowledge"].append(processed_result)
            else:
                result = f"Unknown tool: {tool_name}"

            outputs.append(
                ToolMessage(
                    content=json.dumps(processed_result if 'processed_result' in locals() else result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            print(f"Error processing tool {tool_name}: {e}")
            outputs.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
    return updates


def call_agent_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState],
                   config: RunnableConfig,
                   tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Call the agent with appropriate tools"""
    tools = state["analyst"].tools
    tools = [tools_registry[tool] for tool in tools if tool in tools_registry]
    
    llm = get_llm(tools)
    # Auto-detect placeholders từ system prompt
    base_prompt = state["analyst"].system_prompt
    placeholders = re.findall(r'\{(\w+)\}', base_prompt)
    
    # Prepare available values
    format_values = {
        'question': state.get('question', ''),
        'context': state.get('image_caption', ''),
    }
    
    # Chỉ format với placeholders có trong prompt
    format_dict = {key: format_values[key] for key in placeholders if key in format_values}
    
    formatted_prompt = base_prompt.format(**format_dict)
    
    system_prompt = SystemMessage(content=formatted_prompt)
    question_prompt = HumanMessage(content=f"question: {state['question']}")
    history = state.get("messages", [])
    sequence = [system_prompt, question_prompt] + history

    response = llm.invoke(sequence, config)
    
    updates = dict(state)
    updates["messages"] = state["messages"] + [response]
    
    return updates


def final_reasoning_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> Dict[str, Any]:
    """Final reasoning node to synthesize results"""
    
    print("Processing final reasoning for state:", {k: v for k, v in state.items() if k != "messages"})
    
    # Auto-detect placeholders từ final_system_prompt
    base_prompt = state["analyst"].final_system_prompt
    placeholders = re.findall(r'\{(\w+)\}', base_prompt)
    
    # Prepare available values
    format_values = {
        'context': state.get("image_caption", ""),
        'question': state.get("question", ""),
        'candidates': state.get("answer_candidate", ""),
        'KBs_knowledge': "\n".join(state.get("KBs_Knowledge", [])),
        'LLM_knowledge': state.get("LLM_Knowledge", "")
    }
    
    # Chỉ format với placeholders có trong prompt
    format_dict = {key: format_values[key] for key in placeholders if key in format_values}
    
    final_system_prompt = base_prompt.format(**format_dict)
    
    llm = get_llm(temperature=0.1)
    
    system_msg = SystemMessage(content=final_system_prompt)
    human_msg = HumanMessage(content="Please provide your final analysis and answer.")
    
    final_response = llm.invoke([system_msg, human_msg])
    print("Final answer candidate:", state.get("answer_candidate", None))
    
    return {
        "messages": [final_response],
        "results": [final_response.content],
        "number_of_steps": state.get("number_of_steps", 0) + 1
    }


def should_continue(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> str:
    """Decide whether to continue with tools or move to final reasoning"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If no tool calls, go to final reasoning  
    if not getattr(last_message, "tool_calls", None):
        return "final_reasoning"
    # If has tool calls, continue with tools
    else:
        return "continue"