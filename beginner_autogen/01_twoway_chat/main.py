import autogen


def main():
  config_list = autogen.config_list_from_json(
      env_or_file="OAI_CONFIG_LIST.json"
  )

  assistant = autogen.AssistantAgent(
      name="Assistant",
      llm_config={
          "config_list": config_list
      },
      system_message="You are a travel agent that plans great vacations",
      is_termination_msg=lambda msg: "Thank you" in msg["content"]
  )

  user_proxy = autogen.UserProxyAgent(
      name="user",
      human_input_mode="AlWAYS",
      code_execution_config={
          "work_dir": "coding",
          "use_docker": False,

      }
  )

  user_proxy.initiate_chat(
      assistant, message="Plan me a great sunny vacation"
  )


if __name__ == "__main__":
  main()
