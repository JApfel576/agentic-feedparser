from fastapi import FastAPI
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

# Source - https://stackoverflow.com/a/53496263
# Posted by Orly
# Retrieved 2026-09-24, License - CC BY-SA 4.0

# set up logging to file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./log/myapp.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


app = FastAPI()


def check_url(url_input: str) -> str:
    """Ensure url is proper format"""
    try:
        Website(url=url_input)
    except ValidationError as e:
        print(e)
    return url_input


def convert_to_rss(url_input: str) -> str:
    """Create url from one given for rss feed"""
    site = Website(url=url_input)

    pattern = r"q=site(\:|%3A)(?:%20|\s)?[a-z0-9.-]+(\.com).*"
    if not re.match(pattern, str(site.url.query)):
        raise ValueError("query not expected format for google news site search")

    path_str = "/rss" + site.url.path
    return urlunparse(
        (site.url.scheme, site.url.host, path_str, "", site.url.query, "")
    )


def extract_site(url_input: str) -> str:
    """Format site portion of query to folder name for data"""
    pattern = "[A-Za-z]+.com"
    site = re.findall(pattern, url_input)
    site_fmtd = re.sub("\\.", "_", site[1])
    return site_fmtd


def recent_feed_data(path: str) -> str:
    """Get most recent file name for feed data"""
    file_path = Path(path)
    files = [str(p.resolve()) for p in file_path.iterdir() if p.is_file()]
    files_data = sorted([f for f in files if "state.json" not in f])
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
    (
        """If correct url format is provided"""
        """, return created feed url"""
    )
    return convert_to_rss(url_input)


@app.get("/rss")
def rss_endpoint(url_input: RssUrl):
    """Accepts search url and returns RSS url"""
    return url_input


from fastapi import HTTPException


@app.get("/data", response_model=Model)
def feed_data(url_input: RssUrl) -> Any:
    """Get data using created feed url and store in site specific folder"""
    site_folder = extract_site(url_input)
    out_dir = f"var/data/{site_folder}"

    rss_poller = FeedPoller(url=url_input, out_dir=out_dir)
    result = rss_poller.poll()

    if not result:
        # poll() returned falsy — could mean "not modified" or a real failure.
        # Don't fabricate a fake payload; tell the caller explicitly.
        logging.info(f"poll() returned no new data for {url_input}")
        return {
            "header": {"etag": "", "updated": ""},
            "items": [],
            "status": "no_new_data",  # explicit, not inferred from empty fields
        }

    file = recent_feed_data(path=out_dir)
    if file is None:
        logging.error(f"poll() succeeded but no data file found in {out_dir}")
        raise HTTPException(
            status_code=500, detail="poll succeeded but no output file found"
        )

    try:
        with open(file) as f:
            data = json.load(f)
            data["filename"] = file
        logging.info(f"Loaded feed data from {file}")
        data["status"] = "ok"
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file}")
        raise HTTPException(status_code=500, detail="feed data file not found")
    except json.decoder.JSONDecodeError:
        logging.error(f"File was not valid JSON: {file}")
        raise HTTPException(status_code=500, detail="feed data file is corrupted")
