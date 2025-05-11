import autogen

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

task = """Based on the visual features in the image, how would you describe the woman's appearance in terms of clothing, grooming, and expression?
    <img https://drive.google.com/file/d/1SzRkmcxEE--q0awk7Yfu6dhhO1XzvdAx/view?usp=drive_link>
    """

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
    You are a professional writer, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    You should improve the quality of the content based on the feedback from the user.
    <img ./2cd5720a-58a0-4e25-9aec-32ffea07bc06.png>
    """,
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "my_code",
        "use_docker": False,
    }
)

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""
    You are a critic, known for your thoroughness and commitment to standards.
    Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring
    all materials align with required guidelines.
    """,
)


def reflection_message(recipient, messages, sender, config):
  print("Reflecting...")
  return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


user_proxy.register_nested_chats(
    [
        {
            "recipient": critic,
            "message": reflection_message,
            "summary_method": "last_msg",
            "max_turns": 1
        }
    ],
    trigger=writer
)

user_proxy.initiate_chat(recipient=writer, message=task,
                         max_turns=3, summary_method="last_msg")
