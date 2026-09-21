import json
from pathlib import Path
from random import sample
from curl_cffi import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder
import ftfy

path = "var/data/Anasdaq_com/"
filename = "20260918_001730.json"
file_path = Path(path) / filename
with open(file_path) as f:
    data = json.load(f)


def get_feed_info(data):
    new_dict = {"title": [], "published": [], "link": [], "feed_updated": ""}
    for key in data.keys():
        if key == "header":
            new_dict["feed_updated"] = data[key]["updated"]
        if key == "items":
            for item in data[key]:
                new_dict["title"].append(item["title"])
                new_dict["published"].append(item["published"])
                new_dict["link"].append(item["link"])
            return new_dict


def flat_json_text(new_dict):
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


def processed_data(entries, path, filename):
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "w") as f:
        json.dump(entries, f, indent=4)


def fetch_and_parse_url(actual_url):
    try:
        # Add the impersonate parameter to match a real browser fingerprint
        response = requests.get(actual_url, impersonate="chrome120", timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

        # Get first paragraph text from the fetched page
        min_words = 40  # Minimum number of words in a sentence
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

        # Check for 404, 500, or other HTTP error codes
        if response.status_code != 200:
            print(f"Skipping: Received bad status code ({response.status_code})")
            return

        # If it passes the checks, extract your data
        html_content = response.text
        print("Success: Page successfully grabbed.")

        # ... Your BeautifulSoup parsing logic here ...

    except requests.exceptions.Timeout:
        # Handles instances where the server hangs the connection
        print("Skipping: Connection timed out.")

    except requests.exceptions.RequestException as e:
        # Catch-all for network issues (like DNS failure or dropped sockets)
        print(f"Skipping: Network error occurred -> {e}")


def get_sample_text(entries):
    # Fetch the actual URL from the Google News link using gnewsdecoder
    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
        )
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
            sample_text = fetch_and_parse_url(actual_url)
            return f"This is the sample text for entry {entry['id']}: {sample_text}"


def create_document(path, filename):
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document = []
    with open(processed_path, "r") as f:
        processed_file = json.load(f)
        with open(document_path, "w") as f:
            for entry in processed_file:
                text = f"{entry['text']} {entry.get('sample_text', '')}"
                document.append(text)
            f.write("\n\n".join(document))


new_dict = get_feed_info(data)
flat_items = flat_json_text(new_dict)

entries = [
    {"id": i, "text": t, "link": l}
    for i, t, l in zip(flat_items["id"], flat_items["text"], flat_items["link"])
]

for entry in entries[:10]:  # Limit to first 10 entries for testing
    entry["sample_text"] = get_sample_text(entry)

processed_path = Path(path) / "processed" / filename
processed_data(entries, processed_path, filename)

document_path = Path(path) / "document" / filename
create_document(processed_path, document_path)
