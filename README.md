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

6. Optionally fetch, convert to Markdown, chunk, and embed **TODO**

  

## Why Use It

  

This approach avoids scraping entire sites unless the metadata indicates the article is relevant, making RAG ingestion faster and more efficient.

  

## Project Structure

  

<!-- TREE_START -->
```
.
├── Dockerfile
├── LICENSE
├── README.Docker.md
├── README.md
├── compose.yaml
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
│   │   ├── pyproject.toml
│   │   └── src
│   │       └── feed_agents
│   │           ├── __init__.py
│   │           └── feed_agents.py
│   └── feedpoller
│       ├── pyproject.toml
│       └── src
│           ├── feedpoller
│           │   ├── __init__.py
│           │   └── feedpoller.py
│           └── feedpoller.egg-info
│               ├── PKG-INFO
│               ├── SOURCES.txt
│               ├── dependency_links.txt
│               ├── requires.txt
│               └── top_level.txt
├── pyproject.toml
├── services
│   └── fastapi_app
│       ├── pyproject.toml
│       └── src
│           └── fastapi_app
│               └── main.py
├── uv.lock
└── var
    └── data
        ├── bbc_com
        │   ├── 20260515_183820.json
        │   └── state.json
        └── reuters_com
            ├── 20260515_183846.json
            └── state.json

21 directories, 34 files
```
<!-- TREE_END -->
