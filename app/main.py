from fastapi import FastAPI, Query
from typing import Any
from pydantic import BaseModel, ValidationError, HttpUrl
from urllib.parse import urlunparse
import json
import re


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

app = FastAPI()

# Ensure url is proper format
def check_url(url_input):
  try: 
     Website(url = url_input)
  except ValidationError as e:
    print(e)
  return url_input

# Create url from one given for rss feed
def create_url(url_input):
  site = Website(url = url_input)
  path_str = "/rss" + site.url.path
  url_str = urlunparse((site.url.scheme
                    , site.url.host
                    , path_str
                    , ""
                    , site.url.query
                    , ""))
  q_pattern = r"q=site(\:|%3A)(?:%20|\s)?[a-z0-9.-]+\.com"
  if not re.match(q_pattern, str(site.url.query)):
    return "query not expected format for google news site search"
  return url_str
     
  
# Check app health
@app.get("/health")
def health_check():
  return {"status": "ok"}

# If correct url format is provided, return created feed url
@app.get("/url")
def url_provider(url_input: str | None = None) -> str:
  try:
     check_url(url_input)
     return create_url(url_input) 
  except ValueError as e:
     return e

# Get data using created feed url
@app.get("/data", response_model=Model)
def feed_data(url_input: str | None = None) -> Any:
  data = {
    "header":{"etag":""
              , "updated":""}
    , "items": [{"title":"test"
              , "summary":""
              , "published":""
              , "guid":""
              , "link":""}]
              }
  return data