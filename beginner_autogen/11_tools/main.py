import autogen
from typing import Annotated
import datetime


config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["qwen2-vl-2b-instruct"]
    }
)

llm_config = {
    "cache_seed": 43,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,  # in seconds
}

def get_weather(location: Annotated[str, "The location"]) -> str:
  if location == "Florida":
    return "It's hot in Florida!"
  elif location == "Maine":
    return "It's cold in Maine"
  else:
    return f"I don't know this place {location}"


def get_time(timezone: Annotated[str, "The timezone we are in"]) -> str:
  now = datetime.datetime.now()

  # Get the time with timezone information
  current_time = now.strftime("%Y-%m-%d %H:%M:%S %Z")

  return f"Current time in your {timezone}: {current_time}"


assistant = autogen.ConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant. "
    "Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
    code_execution_config=False,
    max_consecutive_auto_reply=4
)

user_proxy = autogen.ConversableAgent(
    name="User",
    is_termination_msg=lambda msg:
        (msg.get("content") or "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=4,
    human_input_mode="NEVER"
)

# Register the tool signature with the assistant agent.
assistant.register_for_llm(
    name="get_weather", description="Get the current weather for a specific location")(get_weather)
assistant.register_for_llm(
    name="get_time", description="The IANA time zone name, e.g. America/Los_Angeles")(get_time)


# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="get_weather")(get_weather)
user_proxy.register_for_execution(name="get_time")(get_time)

user_proxy.initiate_chat(
    assistant,
    message="What weather is it in Florida?"
)
