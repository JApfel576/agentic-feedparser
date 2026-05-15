from fastapi import FastAPI, Query
from typing import Any, Annotated
from pydantic import BaseModel, ValidationError, HttpUrl, AfterValidator
from urllib.parse import urlunparse
import json
import re
from feedpoller import FeedPoller 
from pathlib import Path
import logging


class Website(BaseModel):
   url: HttpUrl


class Header(BaseModel):
    etag: str | None = None
    updated: str


class Item(BaseModel):
    title: str
    summary: str
    published: str | None = None
    guid: str | None = None
    link: str


class Model(BaseModel):
    header: Header
    items: list[Item]


logger = logging.getLogger(__name__)
app = FastAPI()


def check_url(url_input:str) -> str:
  """Ensure url is proper format"""
  try: 
     Website(url = url_input)
  except ValidationError as e:
    print(e)
  return url_input


def convert_to_rss(url_input: str) -> str:
  """Create url from one given for rss feed"""
  site = Website(url = url_input)

  pattern = r"q=site(\:|%3A)(?:%20|\s)?[a-z0-9.-]+\.com"
  if not re.match(pattern, str(site.url.query)):
    raise ValueError ("query not expected format for google news site search")
  
  path_str = "/rss" + site.url.path
  return urlunparse((site.url.scheme
                    , site.url.host
                    , path_str
                    , ""
                    , site.url.query
                    , ""))


def recent_feed_data(path: str = "var/data") -> str:
  """Get most recent file name for feed data"""
  file_path = Path(path)
  files = [
     str(p.resolve()) for p in file_path.iterdir() if p.is_file()
     ]
  files_data = sorted([
     f for f in files if 'state.json' not in f
     ])
  if not files_data:
     return None
  return files_data[-1]
  

SearchUrl = Annotated[str, AfterValidator(check_url)]
RssUrl = Annotated[str, AfterValidator(convert_to_rss)]


@app.get("/health")
def health_check():
  """Check app health"""
  return {"status": "ok"}


@app.get("/url")
def url_provider(url_input: SearchUrl):
  """If correct url format is provided""" \
  """, return created feed url"""
  return convert_to_rss(url_input)


@app.get("/rss")
def rss_endpoint(url_input: RssUrl):
   """Accepts search url and returns RSS url"""
   return url_input


@app.get("/data", response_model=Model)
def feed_data(url_input: RssUrl) -> Any:
  """Get data using created feed url"""
  rss_poller = FeedPoller(url_input)
  result = rss_poller.poll()
  if result:
     file = recent_feed_data()
     print(file)
     try:
        with open(file, 'r') as f:
           data = json.load(f)
        logger.info("Loaded feed data from file") 
        return data
     except FileNotFoundError:
        logger.error("File was not found")
     except json.decoder.JSONDecodeError:
        logger.error("File was not serialized")