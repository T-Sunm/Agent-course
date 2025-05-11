from autogen import ConversableAgent

# Cấu hình Qwen2-VL (qua LM Studio API)
qwenvl_config = {
    "config_list": [
        {
            "model": "lmstudio-community/Qwen2-VL-2B-Instruct-GGUF",
            "base_url": "http://127.0.0.1:1234/v1",  # Đổi IP/port nếu cần
            "api_key": "lm-studio",
        }
    ]
}

# Agent 1 - Qwen đóng vai Jack
jack = ConversableAgent(
    name="Jack",
    llm_config=qwenvl_config,
    system_message="Your name is Jack and you are a comedian in a two-person comedy show.",
)

# Agent 2 - Qwen đóng vai Emma
emma = ConversableAgent(
    name="Emma",
    llm_config=qwenvl_config,
    system_message="Your name is Emma and you are a comedian in a two-person comedy show.",
)

# Bắt đầu hội thoại: Jack khơi chuyện với Emma
chat_result = jack.initiate_chat(
    recipient=emma,
    message="Emma, tell me a joke.",
    max_turns=2
)

# In kết quả toàn bộ hội thoại
print("Full Conversation:\n", chat_result)
