import json
from pathlib import Path
from curl_cffi import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder
import ftfy
import os
import logging
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from collections import Counter


class ProcessData:
    def __init__(
        self,
        file_path,
        logger_name: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.file_path = file_path
        self.path = os.path.dirname(file_path)
        self.filename = os.path.basename(file_path)
        self.processed_path = Path(self.path) / "processed" / self.filename
        self.document_path = (
            Path(self.path) / "document" / self.filename.replace(".json", ".txt")
        )
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        self.document_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger_name = logger_name
        self.logger = logger or logging.getLogger(logger_name)

    def _parse_published(self, value: str) -> datetime | None:
        """Parse RFC 822 or ISO 8601 to a UTC-aware datetime. None if unparseable."""
        if not value:
            return None
        for parse in (parsedate_to_datetime, datetime.fromisoformat):
            try:
                dt = parse(value)
            except (TypeError, ValueError):
                continue
            if dt.tzinfo is None:
                dt = dt.replace(
                    tzinfo=timezone.utc
                )  # assume UTC when no offset is given
            return dt.astimezone(timezone.utc)
        return None

    def latest_items(self, data: dict, n: int = 5) -> dict:
        """Return a copy of the feed object keeping only the n most recent items."""
        items = data.get("items") or []
        dated = [(self._parse_published(i.get("published", "")), i) for i in items]
        dated.sort(
            key=lambda p: p[0] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return {**data, "items": [i for _, i in dated[:n]]}

    def get_feed_info(self, data):
        items = data.get("items") or []
        return {
            "feed_updated": (data.get("header") or {}).get("updated", ""),
            "title": [i.get("title", "") for i in items],
            "published": [i.get("published", "") for i in items],
            "published_fmtd": [
                self._parse_published(i.get("published", "")) for i in items
            ],
            "link": [i.get("link", "") for i in items],
        }

    def flat_json_text(self, new_dict):
        flat_dict = {"id": [], "text": [], "feed_updated": "", "link": []}
        title_text = []
        published_datetimes = []
        # retrieved the title and published datetime from the new_dict and created a flat text for each item
        for key, values in new_dict.items():
            if key == "title":
                for value in values:
                    title = f"This item is titled {value}"
                    title_text.append(title)
            if key == "published":
                for value in values:
                    published = f"published datetime is {value}"
                    published_datetimes.append(published)
                # create a list of tuples containing the title and published datetime for each item
                joined_info = list(zip(title_text, published_datetimes))
                flat_text = [f"{i[0]}, and {i[1]}" for i in joined_info]
                flat_dict["text"] = flat_text
                flat_dict["feed_updated"] = new_dict["feed_updated"]
                flat_dict["link"] = new_dict["link"]
                # create a list of ids for each item in the flat_text list
                for i in range(len(flat_text)):
                    flat_dict["id"].append(i)
                return flat_dict

    def processed_data(self, entries):
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=4, ensure_ascii=False, default=str)

    def fetch_and_parse_url(self, actual_url):
        try:
            # Add the impersonate parameter to match a real browser fingerprint
            response = requests.get(actual_url, impersonate="chrome120", timeout=15)
            # Check for 404, 500, or other HTTP error codes
            if response.status_code != 200:
                print(f"Skipping: Received bad status code ({response.status_code})")
                return

            if response.status_code == 200:
                soup = BeautifulSoup(
                    response.content.decode("utf-8", errors="ignore"), "html.parser"
                )

            # If it passes the checks, extract your data
            html_content = response.text
            print("Success: Page successfully grabbed.")

            # ...BeautifulSoup parsing logic here ...
            # Get first paragraph text from the fetched page
            min_words = 25  # Minimum number of words in a sentence
            best_word_count = 0
            best_paragraph = None

            for p in soup.find_all("p"):
                raw_text = p.get_text(strip=True)
                text = ftfy.fix_text(raw_text)  # Fix text encoding issues
                if not text:
                    continue  # Skip empty paragraphs

                word_count = len(text.split())
                if word_count < min_words:
                    continue  # Skip paragraphs with fewer than min_words
                if word_count > best_word_count:
                    best_word_count = word_count
                    best_paragraph = text

            if best_paragraph is None:
                print("No qualifying paragraph found.")
                return None

            return best_paragraph.strip()

        except requests.exceptions.Timeout:
            # Handles instances where the server hangs the connection
            print("Skipping: Connection timed out.")

        except requests.exceptions.RequestException as e:
            # Catch-all for network issues (like DNS failure or dropped sockets)
            print(f"Skipping: Network error occurred -> {e}")

    def get_sample_text(self, entry):
        # Fetch the actual URL from the Google News link using gnewsdecoder
        try:
            decoded_data = gnewsdecoder(entry["link"])
            if decoded_data.get("status"):
                print("Decoded URL:", decoded_data["decoded_url"])
            else:
                print("Error:", decoded_data["message"])
        except Exception as e:
            print(f"Error occurred: {e}")
        # Check if the decoded_url is a dictionary and contains the "decoded_url" key
        if isinstance(decoded_data, dict) and decoded_data.get("decoded_url"):
            actual_url = decoded_data.get("decoded_url")
            print(f"Processing: {actual_url}")
            if not actual_url.endswith(".pdf"):
                sample_text = self.fetch_and_parse_url(actual_url)
            else:
                sample_text = "This is placeholder sample text for pdf content."  # Placeholder for PDF or non-HTML content
            return f"This is the sample text for entry {entry['id']}: {sample_text}"

    def create_document(self):
        self.document_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.processed_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        print(
            Counter(
                "missing"
                if "sample_text" not in e
                else "none"
                if e["sample_text"] is None
                else "empty"
                if not str(e["sample_text"]).strip()
                else "ok"
                for e in entries
            )
        )

        chunks = []
        for entry in entries:
            text = (entry.get("text") or "").strip()
            sample = (entry.get("sample_text") or "").strip()
            combined = f"{text}\n{sample}".strip()
            if combined:
                chunks.append(combined)
            else:
                self.logger.warning("Skipping empty entry: %s", entry.get("id"))

        with open(self.document_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(chunks) + "\n")

    def run(self) -> dict:
        result = {"ok": False, "error": None, "entries": 0, "sample_failures": []}

        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.exception("could not load %s", self.file_path)
            result["error"] = f"load failed: {type(exc).__name__}: {exc}"
            return result

        try:
            trimmed_data = self.latest_items(data, n=10)
            new_dict = self.get_feed_info(trimmed_data)
            flat_items = self.flat_json_text(new_dict)
            entries = [
                {"id": i, "text": t, "link": link}
                for i, t, link in zip(
                    flat_items["id"], flat_items["text"], flat_items["link"]
                )
            ]
        except (KeyError, TypeError, AttributeError) as exc:
            self.logger.exception("unexpected feed structure")
            result["error"] = f"transform failed: {type(exc).__name__}: {exc}"
            return result

        result["entries"] = len(entries)
        if not entries:
            result["error"] = "no entries found"
            return result

        # Write 1: persist unenriched entries (safety net)
        try:
            self.processed_data(entries)
        except OSError as exc:
            self.logger.exception("failed writing processed data")
            result["error"] = f"write failed: {type(exc).__name__}: {exc}"
            return result

        # Enrich
        for entry in entries:
            try:
                sample = self.get_sample_text(entry)
            except Exception as exc:
                self.logger.exception(
                    "get_sample_text failed for id=%s", entry.get("id")
                )
                sample = None
                result["sample_failures"].append(
                    {"id": entry.get("id"), "error": f"{type(exc).__name__}: {exc}"}
                )
            else:
                if not sample:
                    result["sample_failures"].append(
                        {"id": entry.get("id"), "error": "no sample text"}
                    )
            entry["sample_text"] = sample

        # Write 2: persist enriched entries, must come after the loop
        try:
            self.processed_data(entries)
        except OSError as exc:
            self.logger.exception("failed writing enriched data")
            result["error"] = f"enriched write failed: {type(exc).__name__}: {exc}"
            return result

        try:
            self.create_document()
        except Exception as exc:
            self.logger.exception("failed creating document")
            result["error"] = f"document failed: {type(exc).__name__}: {exc}"
            return result

        result["ok"] = True
        result["document_path"] = str(self.document_path)
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("process_data")
    file_path = "var/data/Anasdaq_com/20260918_001730.json"
    report = ProcessData(file_path=file_path, logger=logger).run()

    if not report or not report.get("ok"):
        logger.error(
            "process_file failed for %s: %s",
            file_path,
            (report or {}).get("error", "no report returned"),
        )

    logger.info("process_file wrote %s", report)
