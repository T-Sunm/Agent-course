# CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2.5-Coder-3B-Instruct" \
#   --port 8000 \
#   --trust-remote-code \
#   --dtype auto \
#   --gpu-memory-utilization 0.40

import os
import torch

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def fetch_latest_news_titles_and_urls(url: str) -> list[tuple[str, str]]:
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
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

from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent, register_function
import autogen

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["Qwen/Qwen2.5-Coder-3B-Instruct"]
    }
)

llm_config = {
    "cache_seed": 43,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,  # in seconds
}

executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    functions=[
        fetch_latest_news_titles_and_urls,
        extract_news_article_content,
        summarize_news,
        classify_topic,
    ],
)

# ──────────────────── AGENTS ────────────────────
# (1) Một agent duy nhất: vừa viết vừa chạy code
writer_agent = AssistantAgent(
    name="writer_agent",
    system_message=(
        "Bạn là một nhà báo. Viết một bài báo ngắn tiếng Việt (~200 từ) về bất kỳ chủ đề nào bạn chọn. "
        "Sau khi viết xong, hãy in nội dung bài báo ra màn hình và kết thúc bằng dòng TERMINATE."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# (2) Agent xử lý kết quả
result_handler = AssistantAgent(
    name="result_handler",
    system_message=(
        "Bạn nhận output của script, có thể gọi các hàm summarize_news hoặc classify_topic "
        "để tóm tắt. Kết thúc bằng dòng TERMINATE."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Cho phép result_handler gọi 2 tool thông qua coding_agent
register_function(summarize_news, caller=result_handler, executor=writer_agent, description="Tóm tắt tin tức")
register_function(classify_topic, caller=result_handler, executor=writer_agent, description="Phân loại chủ đề tin tức")

# ──────────────────── USER PROXY ────────────────────
user_proxy = autogen.UserProxyAgent(name="user", 
                              human_input_mode="NEVER", 
                              code_execution_config={
                                    "work_dir": "coding",
                                    "use_docker": False,
                                },
                              is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,)

user_proxy.initiate_chats([
    {
        "recipient": writer_agent,
        "message": (
            "Viết một bài báo ngắn tiếng Việt (~200 từ) về một chủ đề bất kỳ mà bạn chọn. "
            "và kết thúc bằng dòng TERMINATE."
        ),
        "summary_method": "last_msg",
        "clear_history": True
    },
    {
        "recipient": result_handler,
        "message": (
            "Dưới đây là một bài báo. Hãy tóm tắt nội dung bằng hàm summarize_news. "
            "Nếu có thể, hãy dùng classify_topic để xác định bài viết thuộc chủ đề gì "
            "(ví dụ: giáo dục, công nghệ, môi trường...). Kết thúc bằng TERMINATE."
        ),
        "clear_history": False
    }
])