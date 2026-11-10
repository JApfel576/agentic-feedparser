# RSS Feed Poller for Targeted RAG Pipelines

  

This project provides a small RSS feed poller that pulls article metadata from any site indexed by Google News. Instead of scraping full pages upfront, it collects only titles, summaries, and links so an LLM can decide which articles are worth deeper processing.

  

## How It Works

  

1. Poll Google News RSS feeds 
**DONE**

2. Extract minimal metadata (title, summary, link, timestamps) **DONE**

3. Store only new or updated entries 
**DONE**

4. Let an LLM choose which articles to fully scrape 
**IN PROGRESS**
<img src="assets/agent_teams_graph.png" width="600" alt="Screenshot">

6. Optionally fetch, convert to Markdown, chunk, and embed **TODO**

  

## Why Use It

  

This approach avoids scraping entire sites unless the metadata indicates the article is relevant, making RAG ingestion faster and more efficient.

  

## Project Structure

  

<!-- TREE_START -->
```
.
├── AGENTS.md
├── Dockerfile
├── LICENSE
├── README.Docker.md
├── README.md
├── assets
│   └── agent_teams_graph.png
├── compose.yaml
├── ingest.py
├── packages
│   ├── base_agents
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   └── src
│   │       ├── base_agents
│   │       │   ├── __init__.py
│   │       │   └── base_agents.py
│   │       └── base_agents.egg-info
│   │           ├── PKG-INFO
│   │           ├── SOURCES.txt
│   │           ├── dependency_links.txt
│   │           ├── requires.txt
│   │           └── top_level.txt
│   ├── feed_agents
│   │   ├── README.md
│   │   ├── log
│   │   │   └── myapp.log
│   │   ├── pyproject.toml
│   │   └── src
│   │       ├── feed_agents
│   │       │   ├── __init__.py
│   │       │   └── feed_agents.py
│   │       └── var
│   │           └── data
│   │               └── chroma_db_storage
│   │                   └── chroma.sqlite3
│   ├── feedpoller
│   │   ├── pyproject.toml
│   │   └── src
│   │       ├── feedpoller
│   │       │   ├── __init__.py
│   │       │   └── feedpoller.py
│   │       └── feedpoller.egg-info
│   │           ├── PKG-INFO
│   │           ├── SOURCES.txt
│   │           ├── dependency_links.txt
│   │           ├── requires.txt
│   │           └── top_level.txt
│   └── process_data
│       ├── README.md
│       ├── pyproject.toml
│       └── src
│           └── process_data
│               ├── __init__.py
│               ├── ingest.py
│               ├── process_data.py
│               └── query.py
├── pyproject.toml
├── services
│   └── fastapi_app
│       ├── log
│       │   └── myapp.log
│       └── src
│           └── fastapi_app
│               └── main.py
├── uv.lock
└── var
    └── data
        ├── bloomberg_com
        │   ├── 20261105_004715.json
        │   ├── document
        │   │   └── 20261105_004715.txt
        │   ├── processed
        │   │   └── 20261105_004715.json
        │   └── state.json
        ├── briefing_com
        │   ├── 20261105_004712.json
        │   ├── document
        │   │   └── 20261105_004712.txt
        │   ├── processed
        │   │   └── 20261105_004712.json
        │   └── state.json
        ├── chroma_db_storage
        │   ├── 203ae349-b8d2-4eec-89ce-d7d51305968d
        │   │   ├── data_level0.bin
        │   │   ├── header.bin
        │   │   ├── length.bin
        │   │   └── link_lists.bin
        │   └── chroma.sqlite3
        ├── cnbc_com
        │   ├── 20261105_004716.json
        │   ├── document
        │   │   └── 20261105_004716.txt
        │   ├── processed
        │   │   └── 20261105_004716.json
        │   └── state.json
        ├── cnn_com
        │   ├── 20261105_004714.json
        │   ├── document
        │   │   └── 20261105_004714.txt
        │   ├── processed
        │   │   └── 20261105_004714.json
        │   └── state.json
        ├── finance_yahoo_com
        │   ├── 20261105_004716.json
        │   ├── document
        │   │   └── 20261105_004716.txt
        │   ├── processed
        │   │   └── 20261105_004716.json
        │   └── state.json
        └── marketwatch_com
            ├── 20261105_004716.json
            ├── document
            │   └── 20261105_004716.txt
            ├── processed
            │   └── 20261105_004716.json
            └── state.json

48 directories, 70 files
```
<!-- TREE_END -->
