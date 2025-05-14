# filename: scrape_vnexpress_ai.py

import requests
from bs4 import BeautifulSoup

# Step 1: Fetch the latest news titles and URLs related to 'trí tuệ nhân tạo'
url = "https://vnexpress.net"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all news items on the page
news_items = soup.find_all('div', class_='list-news-item')

# Extract titles and URLs
ai_news_titles_and_urls = []
for item in news_items:
    title_tag = item.find('h3', class_='title-news')
    link_tag = title_tag.find('a')
    if title_tag and link_tag:
        title = title_tag.get_text(strip=True)
        url = link_tag['href']
        if 'trí tuệ nhân tạo' in title.lower():
            ai_news_titles_and_urls.append((title, url))

# Output the results
for title, url in ai_news_titles_and_urls:
    print(f"Title: {title}")
    print(f"URL: {url}")
    print("-" * 40)