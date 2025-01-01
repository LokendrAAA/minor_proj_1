import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_article_links(query, max_results=20):
    """
    Fetches links to individual PubMed articles based on a search query.
    Args:
    - query (str): Search term for PubMed.
    - max_results (int): Number of article links to fetch.

    Returns:
    - List of URLs for individual articles.
    """
    article_links = []
    base_url = "https://pubmed.ncbi.nlm.nih.gov"
    
    # Start from the first page
    page_number = 1
    
    while len(article_links) < max_results:
        search_url = f"{base_url}/?term={query}&page={page_number}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find article links on the current page
        links = [base_url + link['href'] for link in soup.find_all('a', class_='docsum-title', limit=max_results - len(article_links))]
        
        article_links.extend(links)
        page_number += 1
        time.sleep(1)  # To avoid overwhelming the server

        if not links:  # No more results found
            break

    return article_links[:max_results]

def fetch_abstract(article_url):
    """
    Fetches the abstract text from an individual article page.
    Args:
    - article_url (str): URL of the article.

    Returns:
    - Abstract text (str) if found, otherwise None.
    """
    response = requests.get(article_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the abstract
    abstract_section = soup.find('div', class_='abstract-content selected')
    if abstract_section:
        return abstract_section.get_text().strip()
    return None

def fetch_pubmed_abstracts(query, max_results=20):
    """
    Fetches abstracts from PubMed based on a search query.
    Args:
    - query (str): Search term for PubMed.
    - max_results (int): Number of abstracts to fetch.

    Returns:
    - DataFrame containing abstracts.
    """
    article_links = fetch_article_links(query, max_results)
    abstracts = []

    for url in article_links:
        print(f"Fetching abstract from: {url}")
        abstract = fetch_abstract(url)
        if abstract:
            abstracts.append({'url': url, 'abstract': abstract})
        time.sleep(1)  # To avoid overwhelming the server

    return pd.DataFrame(abstracts)

def save_abstracts_to_csv(df, filename="abstracts.csv"):
    """
    Saves abstracts to a CSV file. Appends if the file already exists.
    Args:
    - df (DataFrame): DataFrame containing abstracts.
    - filename (str): Name of the CSV file.
    """
    if os.path.exists(filename):
        # Append new data, avoid duplicates by checking URLs
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset='url')
        combined_df.to_csv(filename, index=False)
        print(f"Appended {len(df)} new abstracts to {filename}.")
    else:
        # Save new data if the file doesn't exist
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} abstracts to {filename}.")

# Fetch and save abstracts to CSV
df = fetch_pubmed_abstracts("Image", max_results=20)

if not df.empty:
    save_abstracts_to_csv(df)
else:
    print("No abstracts found.")
