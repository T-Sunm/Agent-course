# filename: scrape_vnexpress_news.sh

#!/bin/bash

# Function to fetch news from VNExpress
fetch_vnexpress_news() {
    local keyword="trí tuệ nhân tạo"
    local url="https://vnexpress.net/search?q=$keyword"

    # Use curl to fetch the page content
    page_content=$(curl -s "$url")

    # Extract news titles and links
    echo "<h1>News about 'Trí tuệ nhân tạo' on VNExpress</h1>"
    echo "<ul>"
    echo "$page_content" | grep -oP '<a href="\K[^"]+">[^<]+</a>' | while read -r link; do
        title=$(echo "$page_content" | grep -oP '<title>\K[^<]+</title>' | head -n 1)
        echo "<li><a href='$link'>$title</a></li>"
    done
    echo "</ul>"
}

# Call the function
fetch_vnexpress_news