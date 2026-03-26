import requests
import feedparser

ITEM_CNT = 15
REUTERS_FINANCE_URL = f"""
https://ir.thomsonreuters.com/rss/news-releases.xml?items={ITEM_CNT}
"""
HEADER_KEYS = ['etag','updated', 'updated_parsed', 'href']
ENTRY_KEYS = ['title', 'summary', 'published', 'published_parsed', 'id'
              , 'link']

# Make request to feed url 
def get_feed(url: str) -> dict:
  try:
    response = requests.get(url=url, timeout=10)
    response.raise_for_status()
    return feedparser.parse(url)
  except requests.RequestException as e:
    print(f"Error fetching feed: {e}")
    return None

# Extract header details
def process_header(feed: dict, key_list: list[str]) -> list:
  new_dict = {k:v for (k,v) in feed.items() if k in key_list}
  return new_dict

# Extract entry details
def process_entries(feed: dict, key_list: list[str]) -> dict:
  processed_entries = [
    {key: entry.get(key) for key in key_list}
    for entry in feed.entries
    ]
  return processed_entries

if __name__ == "__main__":
  feed = get_feed(REUTERS_FINANCE_URL)
  header_dict = process_header(feed, HEADER_KEYS)
  entry_dict = process_entries(feed, ENTRY_KEYS)
