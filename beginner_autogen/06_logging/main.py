import autogen
import autogen.runtime_logging
import json
import pandas as pd

# Vào đây để đọc sqlite: https://inloop.github.io/sqlite-viewer/
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["qwen2.5:1.5b"]
    }
)
llm_config = {
    "cache_seed": 43,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,  # in seconds
}

# Start logging
logging_session_id = autogen.runtime_logging.start(
    config={"dbname": "logging.db"})
print("Started Logging session ID: " + str(logging_session_id))

# Create an agent workflow and run it
assistant = autogen.AssistantAgent(name="assistant", llm_config=llm_config)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

user_proxy.initiate_chat(
    assistant, message="What is the height of the Sears Tower? Only respond with the answer and terminate"
)

# stop logging
autogen.runtime_logging.stop()

# Create function to get log
def get_log(db_name="logging.db", table="chat_completions"):
  """Fetches all rows from the logs table in the SQLite database."""
  import sqlite3

  conn = sqlite3.connect(db_name)
  cursor = conn.cursor()
  query = f"SELECT request, response, cost, start_time, end_time from {table}"
  cursor.execute(query)
  rows = cursor.fetchall()
  column_names = [description[0] for description in cursor.description]
  data = [dict(zip(column_names, row)) for row in rows]

  """
  column_names = ["name", "age"]
  rows = [("Alice", 30), ("Bob", 25)]
  -----> 
  [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
  ]

  """
  conn.close()

  return data

def str_to_dict(s):
  return json.loads(s)


log_data = get_log()

log_data_df = pd.DataFrame(log_data)

log_data_df["total_tokens"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1
)

log_data_df["request"] = log_data_df.apply(
    lambda row: str_to_dict(row["request"])["messages"][0]["content"], axis=1)

log_data_df["response"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1
)

print(log_data_df)
