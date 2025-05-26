# -*- coding: utf-8 -*-
"""Microsoft Learn Scrap with Google Colab.py

# Web Scraping, Processing, and Embedding

## Install necessary libraries
"""

## pip install -q ipywidgets google-colab python-docx pypdf pandas nltk sentence-transformers torch tqdm pyarrow httpx beautifulsoup4 datasets requests

"""## Web scraping and data extraction script
This script crawls a website and extracts text content from each page.

"""

# This script to navigate to the link https://learn.microsoft.com/en-us/ and start web scrapping and data extraction automatically on every page must scrap and extract all data, 100% data

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def is_valid(url):
    """Checks whether `url` is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_all_website_links(url):
    """
    Returns all URLs that is found on `url` in which it belongs to the same website
    """
    urls = set()
    domain_name = urlparse(url).netloc
    try:
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not is_valid(href):
                continue
            if parsed_href.netloc == domain_name:
                urls.add(href)
    except Exception as e:
        print(f"Error processing {url}: {e}")
    return urls

def scrape_page_data(url):
    """Scrapes all text content from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract all text from the page
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def crawl_website(start_url, max_pages=100):
    """Crawls a website and scrapes data from each page."""
    visited_urls = set()
    urls_to_visit = {start_url}
    scraped_data = {}

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue

        print(f"Visiting: {current_url}")
        visited_urls.add(current_url)

        # Scrape data
        data = scrape_page_data(current_url)
        if data:
            scraped_data[current_url] = data

        # Find new links
        new_links = get_all_website_links(current_url)
        for link in new_links:
            if link not in visited_urls:
                urls_to_visit.add(link)

    return scraped_data

# Start the crawling process
start_url = "https://learn.microsoft.com/en-us/"
all_scraped_data = crawl_website(start_url)

# You can now process the `all_scraped_data` dictionary
# For example, print the number of pages scraped and the data from one page:
print(f"\nScraped data from {len(all_scraped_data)} pages.")
if all_scraped_data:
    first_url = list(all_scraped_data.keys())[0]
    print(f"\nData from the first scraped page ({first_url}):")
    # print(all_scraped_data[first_url][:500]) # Print first 500 characters

"""## Data processing and embedding script
This script takes the scraped data, chunks it, and creates embeddings using a sentence transformer model.
"""

# This script to convert, format, embed the full scrapped and extracted data to structured, embedded data chunks

import torch
from sentence_transformers import SentenceTransformer # Changed import
from datasets import Dataset
from tqdm.auto import tqdm

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Splits text into chunks with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - chunk_overlap
        if i >= len(words) - chunk_overlap and i < len(words): # Handle the last chunk
             chunks.append(" ".join(words[i:]))
             break

    return chunks

def process_scraped_data(scraped_data, chunk_size=500, chunk_overlap=50):
    """
    Converts scraped data into formatted chunks and embeds them.
    Returns a list of dictionaries, each containing chunk text, source URL, and embedding.
    """
    processed_chunks = []
    for url, text in tqdm(scraped_data.items(), desc="Processing scraped data"):
        if text:
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk in chunks:
                processed_chunks.append({
                    'text': chunk,
                    'source': url,
                })
    return processed_chunks

def embed_chunks(processed_chunks, model, batch_size=32):
    """Embeds the text chunks using the sentence transformer model."""
    # Extract texts for embedding
    texts_to_embed = [chunk['text'] for chunk in processed_chunks]

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({'text': texts_to_embed})

    # Define a function to apply embeddings
    def get_embeddings(batch):
        return {'embedding': model.encode(batch['text'], convert_to_tensor=True).tolist()}

    # Apply the embedding function in batches
    dataset = dataset.map(get_embeddings, batched=True, batch_size=batch_size)

    # Update the original processed_chunks list with embeddings
    for i, item in enumerate(processed_chunks):
        item['embedding'] = dataset[i]['embedding']

    return processed_chunks

# --- Main script for processing and embedding ---

# Process the scraped data into chunks
formatted_chunks = process_scraped_data(all_scraped_data)

# Embed the chunks
embedded_data = embed_chunks(formatted_chunks, model)

# `embedded_data` is now a list of dictionaries, where each dictionary
# represents a chunk with its text, source URL, and embedding.
# You can now use this data for similarity search, indexing, etc.

print(f"\nCreated {len(embedded_data)} embedded chunks.")
if embedded_data:
    print("\nExample of an embedded chunk:")
embedded_data[0]

"""## Save the embedded dataset to Google Drive
This script saves the processed and embedded data to a JSON file in your Google Drive.

"""

# This script to save all converted, formatted, embedded dataset to the "Output" file on My Drive

import json
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the output file path
output_file_path = '/content/drive/My Drive/Output/embedded_dataset.json'

# Ensure the output directory exists
import os
output_dir = os.path.dirname(output_file_path)
os.makedirs(output_dir, exist_ok=True)

# Save the embedded data to a JSON file
with open(output_file_path, 'w') as f:
    json.dump(embedded_data, f, indent=2)

print(f"\nSaved embedded dataset to: {output_file_path}")
