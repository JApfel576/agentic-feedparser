from pathlib import Path
from dotenv import load_dotenv
import os
import re
from process_data import ProcessData
import logging
from urllib.parse import parse_qs, urlparse, urlunparse
import json

# Resolve the project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

project_root = os.environ["PROJECT_ROOT"]

log_dir = os.path.join(project_root, "packages", "feed_agents", "temp_log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "myapp.log")

# Source - https://stackoverflow.com/a/53496263
# Posted by Orly
# Retrieved 2026-09-24, License - CC BY-SA 4.0

# set up logging to file
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    filename=log_file,
    filemode="w",
)

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger("").addHandler(console)


DATA_DIR = os.path.join(ROOT, "var", "data")
TS_FILE = re.compile(r"^\d{8}_\d{6}\.json$")


_SITE_Q = re.compile(
    r"(?:^|\s)site:(?P<domain>[a-z0-9-]+(?:\.[a-z0-9-]+)+)(?P<path>/\S*)?(?=\s|$)",
    re.IGNORECASE,
)


def _slug(site: str) -> str:
    """'https://www.Reuters.com/world' -> 'reuters_com'"""

    parsed = urlparse(site)
    q = parse_qs(parsed.query).get("q", [""])[0]
    m = _SITE_Q.search(q)
    if not m:
        raise ValueError("query missing a site: filter")
    return m["domain"].lower().removeprefix("www.").replace(".", "_")
    print(q)
    return q


def find_data(site: str) -> str:
    """Return the path of the most recent data file for a site (domain or URL)."""
    site_dir = os.path.join(DATA_DIR, _slug(site))
    if not os.path.isdir(site_dir):
        return f"ERROR: no data directory for '{site}' (looked in {site_dir})"

    files = [f for f in os.listdir(site_dir) if TS_FILE.match(f)]
    if not files:
        return f"ERROR: no timestamped .json files in {site_dir}"

    print(os.path.join(site_dir, max(files)))
    return os.path.join(site_dir, max(files))  # YYYYMMDD_HHMMSS sorts lexically


def process_data(path: str) -> str:
    """Process a data file found by find_response_file and write the output to disk.

    Args:
        path: Absolute path to a .json file under the data directory.

    Returns:
        The output file path on success, or an 'ERROR: ...' message.
    """
    if not path:
        return "ERROR: path is not populated"

    real = os.path.realpath(path)
    root = os.path.realpath(DATA_DIR)
    try:
        if os.path.commonpath([real, root]) != root:
            return f"ERROR: path is outside the data directory: {path}"
    except ValueError:  # different drives on Windows
        return f"ERROR: path is outside the data directory: {path}"
    if not os.path.isfile(real):
        return f"ERROR: file not found: {path}"

    report = ProcessData(file_path=real).run()
    if not report.get("ok"):
        logging.error("process_file failed for %s: %s", real, report.get("error"))

    logging.info("process_file wrote %s", report)
    return f"The file was successfully processed. Output: {report}"

    return report.get("document_path")


# file = os.path.join(DATA_DIR, "reuters_com/20260930_204412.json")

file = find_data("https://news.google.com/search?q=site:reuters.com%20stock%20market")
process_data(file)
