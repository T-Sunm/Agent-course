# CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2.5-Coder-3B-Instruct" \
#   --port 8000 \
#   --trust-remote-code \
#   --dtype auto \
#   --gpu-memory-utilization 0.40

from autogen.coding import LocalCommandLineCodeExecutor
import autogen
from autogen import AssistantAgent, register_function
from autogen import initiate_chats


def fetch_latest_news_titles_and_urls(url: str = "https://vnexpress.net") -> list[tuple[str, str]]:
  """
  This tool extracts the titles and URLs of the latest news articles from a news website's
  homepage.

  Args:
      url: The URL of the news website's homepage.

  Returns:
      list[tuple[str, str]]: A list of titles and URLs of the latest news articles.
  """
  import requests
  from bs4 import BeautifulSoup

  article_urls = []
  article_titles = []
  navigation_urls = []

  response = requests.get(url)
  soup = BeautifulSoup(response.text, "html.parser")

  navigation_bar = soup.find("nav", class_="main-nav")
  if navigation_bar:
    for header in navigation_bar.ul.find_all("li")[2:7]:
      navigation_urls.append(url + header.a["href"])

  for section_url in navigation_urls:
    response = requests.get(section_url)
    section_soup = BeautifulSoup(response.text, "html.parser")
    for article in section_soup.find_all("article"):
      title_tag = article.find("h3", class_="title-news")
      if title_tag:
        title = title_tag.text.strip()
        article_url = article.find("a")["href"]
        article_titles.append(title)
        article_urls.append(article_url)

  return list(zip(article_titles, article_urls))


def extract_news_article_content(url: str) -> str:
  """
  This tool extracts the content of a news article from its URL.

  Args:
      url (str): The URL of the news article.

  Returns:
      str: The content of the news article.
  """
  import requests
  from bs4 import BeautifulSoup

  response = requests.get(url)
  soup = BeautifulSoup(response.text, "html.parser")
  content = ""
  for paragraph in soup.find_all("p"):
    content += paragraph.get_text().strip() + " "
  return content


def summarize_news(text: str) -> str:
  """
  This tool summarizes the given Vietnamese news text.

  Args:
      text (str): The Vietnamese news text to be summarized.

  Returns:
      str: The summarized version of the input text.
  """
  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_name = "VietAI/vit5-base-vietnews-summarization"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(
      model_name, torch_dtype=torch.bfloat16)
  model.to(device)

  formatted_text = "vietnews: " + text + " </s>"

  encoding = tokenizer(formatted_text, return_tensors="pt")
  input_ids = encoding["input_ids"].to(device)
  attention_mask = encoding["attention_mask"].to(device)

  with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,
    )
  summary = tokenizer.decode(
      outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
  )
  return summary


def classify_topic(text: str, topic: str) -> bool:
  """
  This tool classifies whether the given Vietnamese text is related to the specified topic.

  Args:
      text: The Vietnamese text to be classified.
      topic: The string representing the topic to be checked.

  Returns:
      bool: True if the text is related to the topic; False otherwise.
  """
  from transformers import pipeline
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"
  classifier = pipeline(
      "zero-shot-classification",
      model="vicgalle/xlm-roberta-large-xnli-anli",
      device=device,
      trust_remote_code=True,
  )

  candidate_labels = [topic, f"không liên quan {topic}"]
  result = classifier(text, candidate_labels)
  predicted_label = result["labels"][0]

  return predicted_label == topic

def count_chars(text: str) -> int:
  """
  Đếm số ký tự trong văn bản đầu vào.
  Args:
    text: Chuỗi cần đếm.
  Returns:
    Số ký tự (len).
  """
  return len(text)


config_list_llama = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    filter_dict={"model": ["llama-3.2-3b-instruct"]}
)


llm_config_llama = {
    # "cache_seed": 43,
    "temperature": 0,
    "config_list": config_list_llama,
    "timeout": 120,
}

# executor = LocalCommandLineCodeExecutor(
#     timeout=60,
#     work_dir="coding",
#     functions=[
#         fetch_latest_news_titles_and_urls,
#         extract_news_article_content,
#         summarize_news,
#         classify_topic,
#         count_chars
#     ],
# )

# ──────────────────── AGENTS ────────────────────
# (1) Một agent duy nhất: vừa viết vừa chạy code
writer_agent = AssistantAgent(
    name="writer_agent",
    system_message=(
        "You are a Vietnamese news journalist. "
        "You have access to tools like 'fetch_latest_news_titles_and_urls' and 'extract_news_article_content'. "
        "You should try using this tool first to get real articles. "
        "If the tool fails (e.g., due to connection errors, invalid HTML, or returns an empty list) "
        "or if no articles are relevant to the user's topic, you can create 3 short **fictional** articles about 'artificial intelligence'. "
        "Each article should include a title and 3–4 lines of content. "
        "When you're done, reply with 'TERMINATE'."
    ),
    llm_config=llm_config_llama,
    human_input_mode="NEVER",
)

# (2) Agent xử lý kết quả
result_handler = AssistantAgent(
    name="result_handler",
    system_message=(
        "You are a text processing expert. "
        "You have access to tools that can help you complete tasks such as counting characters, classifying topics, or summarizing news. "
        "Whenever appropriate, consider using tools like `count_chars`, `classify_topic`, or `summarize_news` to improve accuracy. "
        "You can respond directly when confident, or use tools if needed. "
        "End your reply with 'TERMINATE' when you're done."
    ),
    llm_config=llm_config_llama,
    human_input_mode="TERMINATE",
    is_termination_msg=lambda msg: (
        (getattr(msg, "content", "") or "").strip().endswith("TERMINATE")
    )
)

# ──────────────────── USER PROXY ────────────────────
user_proxy = autogen.UserProxyAgent(name="user",
                                    human_input_mode="NEVER",
                                    code_execution_config={
                                        "work_dir": "coding",
                                        "use_docker": False,
                                    })


tools_4_writer_agent = [
    (fetch_latest_news_titles_and_urls,
     "Fetch a list of the latest article titles and URLs from a news homepage"),
    (extract_news_article_content,
     "Extract the full content of a news article from its URL"),
]

tools_4_result_handler = [
    (summarize_news, "Summarize a Vietnamese news article"),
    (classify_topic, "Classify whether a given article is related to a specific topic"),
    (count_chars, "Count the number of characters in a piece of text"),
]


for fn, desc in tools_4_writer_agent:
  register_function(
      fn,
      caller=writer_agent,
      executor=user_proxy,
      description=desc
  )
for fn, desc in tools_4_result_handler:
  register_function(
      fn,
      caller=result_handler,
      executor=user_proxy,
      description=desc
  )

chat_results = initiate_chats([
    {
        "sender": user_proxy,
        "recipient": writer_agent,
        "message": (
            "Give me some news articles about artificial intelligence in technology"
        ),
        "summary_method": "last_msg",
        "max_turns": 3,
    },
    {
        "sender": user_proxy,
        "recipient": result_handler,
        "message": (
            "For each article below, count the number of characters and determine if it's about 'artificial intelligence'"
        ),
        "max_turns": 2,
    }
])

for chat_result in chat_results:
  print(chat_result.summary)
  print("\n")
