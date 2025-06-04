"""
ArXiv Paper Fetcher Module
--------------------------
This module provides functionality to fetch academic papers from the arXiv API.
It allows fetching papers based on specific queries and date ranges.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

import arxiv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_papers(query: str = "hep-ex", max_papers: int = 500) -> List[Dict[str, Any]]:
    """
    Fetches papers from arXiv based on a query.
    
    Args:
        query: Search query string in arXiv format
        max_papers: Maximum number of papers to retrieve
        
    Returns:
        List of dictionaries containing paper information
    """
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Create client object
    client = arxiv.Client()
    
    results = client.results(search)
    
    papers = []
    for paper in results:
        papers.append({
            "title": paper.title,
            "summary": paper.summary,
            "submission_date": paper.published.date(),
            "id": paper.entry_id,
            "author": paper.authors,
            "primary_category": paper.primary_category,
            "categories": paper.categories
        })
    
    logger.info(f"Retrieved {len(papers)} papers matching query: '{query}'")
    return papers


def fetch_papers_by_years(
    start_year: int = 2012, 
    end_year: int = 2018, 
    domain: str = 'hep-ex', 
    file_path: str = "./data"
) -> pd.DataFrame:
    """
    Fetches arXiv papers from a range of years and saves them as a DataFrame.
    
    Args:
        start_year: First year to include in search
        end_year: Last year to include in search
        domain: arXiv domain/category to search
        file_path: Directory path to save/load data
        
    Returns:
        DataFrame containing paper information
    """
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)
    
    file_name = f"{file_path}/arxiv_{domain}_papers_{start_year}-{end_year}.parquet"
    logger.info(f"Looking for data in: {file_name} for domain {domain}")
    
    # Check if the file already exists
    if os.path.exists(file_name):
        logger.info(f"Loading existing papers from {file_name}")
        df = pd.read_parquet(file_name, engine='pyarrow')
        return df
        
    # If it doesn't exist, fetch papers using the arXiv API
    logger.info("Fetching papers from API...")
    
    #years = list(range(start_year, end_year + 1))
    years = list(range(start_year, end_year))
    months = [f"{i:02}" for i in range(1, 13)]  # 01, 02, ..., 12

    papers = []
    
    for year in years:
        for month in months:
            query = f"{domain} AND submittedDate:[{year}{month}01 TO {year}{month}31]"
            try:
                month_papers = fetch_papers(query=query, max_papers=500)
                papers.extend(month_papers)
                logger.info(f"Fetched {len(month_papers)} papers for {year}-{month}")
                
                # Avoid hitting API limits
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error fetching papers for {year}-{month}: {e}")
    
    logger.info(f"Total papers retrieved: {len(papers)}")

    if not papers:
        logger.warning("No papers retrieved. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(papers)
    
    # Extract author names from Author objects
    df['author'] = df['author'].apply(lambda x: [author.name for author in x])
    
    # Save as Parquet file
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_parquet(file_name, engine='pyarrow')
    logger.info(f"Saved papers to {file_name}")
    
    return df
