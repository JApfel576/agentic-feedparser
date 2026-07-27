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
<!-- TREE_END -->