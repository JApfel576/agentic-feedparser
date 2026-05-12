import requests
import feedparser
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import platform
import json


HEADER_KEYS = ['etag','updated', 'updated_parsed', 'href']
ENTRY_KEYS = ['title', 'summary', 'published', 'published_parsed', 'guid', 'link']


class FeedPoller():
  def __init__(self, url, out_dir="/var/data"):
    self.url = url
    self.header_keys = HEADER_KEYS
    self.entry_keys = ENTRY_KEYS
    self.out_dir = out_dir
    Path(self.out_dir).mkdir(parents=True, exist_ok=True)
    # Store last seen values. Update to retrieve
    self.last_etag = None 
    self.last_modified = None
    self.last_hash = None
    self.load_state()

# -- Load state of feed based on file ids
  def load_state(self):
    state_file = f"{self.out_dir}/state.json"
    if Path(state_file).exists():
        ids = json.loads(Path(state_file).read_text())
        self.last_etag = ids.get("etag")
        self.last_modified = ids.get("updated")
        self.last_hash = ids.get("hash")
    else:
    # -- create state file if it doesn't exist
      ids = {"etag": "", "updated": "", "hash": ""}
      self.save_state(ids)

  # --- Fetch feed with conditional GET --- 
  def fetch(self):
    headers = {}
    if self.last_etag != "":
      headers["If-None-Match"] = self.last_etag
    if self.last_modified != "":
      headers["If-Modified-Since"] = self.last_modified
    try:
      r = requests.get(self.url, headers=headers, timeout=10)
      if r.status_code == 304:
        return None  # unchanged
      r.raise_for_status()
      return feedparser.parse(r.content)
    except requests.RequestException as e:
      print(f"Error fetching feed: {e}")
      return "error"

  
# --- Extract identifiers ---
  def identifiers(self, feed):
      etag = feed.get("etag")
      updated = feed.feed.get("updated")
      # stable fingerprint from GUIDs
      h = hashlib.sha256()
      for e in feed.entries:
          h.update("".join(json.dumps(e, sort_keys=True, default=str) for e in feed.entries).encode())
      return {
          "etag": etag,
          "updated": updated,
          "hash": h.hexdigest(),
      }

# --- Detect change ---
  def has_changed(self, ids):
    if self.last_etag != "" and ids["etag"] != self.last_etag:
        return True
    if self.last_modified != "" and ids["updated"] != self.last_modified:
        return True
    if self.last_hash != "" and ids["hash"] != self.last_hash:
        return True
    return False
  
 # --- Persist feed ---
  def save(self, feed):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = f"{self.out_dir}/{ts}.json"
        data = {
            "header": {
                "etag": feed.get("etag"),
                "updated": feed.feed.get("updated"),
            },
            "items": [
                {
                    "title": e.get("title"),
                    "summary": e.get("summary"),
                    "published": e.get("published"),
                    "guid": e.get("guid"),
                    "link": e.get("link"),
                }
                for e in feed.entries
            ],
        } 
        Path(path).write_text(json.dumps(data, indent=2))
        return data
  
  def save_state(self, ids):
    state_file = f"{self.out_dir}/state.json"
    Path(state_file).write_text(json.dumps(ids))

  # --- Main poll step ---
  def poll(self):
    feed = self.fetch()
    if feed == "error":
      print("Network error — skipping change detection")
      return print("error")
    if feed is None:
      return False #304 Not Modified
    ids = self.identifiers(feed)
    changed = self.has_changed(ids)
    if changed:
        self.save(feed)
        self.save_state(ids)
    return changed