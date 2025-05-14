# filename: fetch_news.py

import requests
from bs4 import BeautifulSoup
from functions import fetch_latest_news_titles_and_urls, extract_news_article_content, summarize_news

def fetch_news(url: str, max_articles=10) -> list[tuple[str, str]]:
    """
    This tool fetches the latest news titles and URLs from a news website's homepage.

    Args:
        url: The URL of the news website's homepage.
        max_articles: The maximum number of articles to fetch.

    Returns:
        list[tuple[str, str]]: A list of titles and URLs of the latest news articles.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('div', {'class': 'item-news'})
        
        news_items = []
        for article in articles[:max_articles]:
            title = article.find('h3').text.strip()
            link = article.find('a')['href']
            news_items.append((title, link))
        
        return news_items
    except requests.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

def main():
    url = "https://vnexpress.net"
    topic = "trí tuệ nhân tạo"
    max_articles = 10

    # Fetch the latest news titles and URLs
    news_items = fetch_news(url, max_articles)

    # Filter news items related to the specified topic
    filtered_news = [item for item in news_items if classify_topic(item[1], topic)]

    # Print the filtered news items
    for title, url in filtered_news:
        print(f"Title: {title}")
        print(f"URL: {url}")

        # Fetch the content of the news article
        html = fetch_news(url)
        article_content = extract_news_article_content(html)

        # Summarize the news article content
        summary = summarize_news(article_content)

        # Print the summary
        print("Summary:")
        print(summary)
        print("\n")

if __name__ == "__main__":
    main()