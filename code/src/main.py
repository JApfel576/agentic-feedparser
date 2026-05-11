from feedpoller import FeedPoller 
import json

REUTERS_URL = ("https://news.google.com/rss/search"
"q=site%3A%20reuters.com&hl=en-US&gl=US&ceid=US%3Aen"
)

def main(url: str = REUTERS_URL) -> dict[str, bool]:
  poller = FeedPoller(url)
  changed = poller.poll()
  if changed:
      print("Feed changed — new file saved")
      return {"Feed changed": True}
  print("No change")
  return {"Feed changed": False}

if __name__ == "__main__":
    main()